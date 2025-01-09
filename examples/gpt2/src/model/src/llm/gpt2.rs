use candle::{D, DType, Device, Tensor, Result};
//use candle_nn::{Embedding, embedding, Conv1d, conv1d, Linear, VarBuilder, Module, Dropout};
use candle_nn::{Embedding, embedding, Linear, VarBuilder, Module};
use std::collections::HashSet;


#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_epsilon: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub layer_caches: Vec<Option<(Tensor, Tensor)>>,
    pub enabled: bool,
    pub current_position: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, enabled: bool) -> Self {
        Self {
            layer_caches: vec![None; num_layers],
            enabled,
            current_position: 0,
        }
    }

    pub fn get_layer_cache(&self, layer_idx: usize) -> Option<&(Tensor, Tensor)> {
        self.layer_caches[layer_idx].as_ref()
    }

    pub fn set_layer_cache(&mut self, layer_idx: usize, key_value: (Tensor, Tensor)) {
        self.layer_caches[layer_idx] = Some(key_value);
    }

    pub fn get_current_position(&self) -> usize {
        self.current_position
    }

    pub fn set_current_position(&mut self, position: usize) {
        self.current_position = position;
    }

    pub fn clear(&mut self) {
        for cache in &mut self.layer_caches {
            *cache = None;
        }
        self.current_position = 0;  // Reset position when clearing cache
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

pub trait MaskCache {
    fn get_mask(&self, size: usize, batch_size: usize) -> Result<Tensor>;
    fn cached_sizes(&self) -> &HashSet<usize>;
}


#[derive(Debug, Clone)]
pub struct GPT2 {
    wte: Embedding,
    wpe: Embedding,
    h: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}


#[derive(Debug, Clone)]
struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: MLP,
}


#[derive(Debug, Clone)]
struct Attention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}


#[derive(Debug, Clone)]
struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}


impl GPT2 {

    pub fn forward(&self, input_ids: &Tensor, kv_cache: &mut KVCache, mask_cache: Option<&dyn MaskCache>) -> Result<Tensor> {

        let start_pos = if kv_cache.is_enabled() {
            // Get the current position from the cache
            kv_cache.get_current_position()
        } else {
            0
        };

        let (_b_sz, seq_len) = input_ids.dims2()?;
        let positions = Tensor::arange(
            start_pos as u32,
            (start_pos + seq_len) as u32,
            input_ids.device()
        )?;
        let inputs_embeds = self.wte.forward(input_ids)?;


        let position_embeds = self.wpe.forward(&positions)?.unsqueeze(0)?;
        let mut hidden_states = (inputs_embeds + position_embeds)?;

        // Update the cache's position
        if kv_cache.is_enabled() {
            kv_cache.set_current_position(start_pos + seq_len);
        }

        for (layer_idx, block) in self.h.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, kv_cache, mask_cache, layer_idx)?;
        }

        let hidden_states = self.ln_f.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok(logits)
    }



    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {

        let wte_weights = vb.pp("wte").get((config.vocab_size, config.n_embd), "weight")?;
        let wte = Embedding::new(wte_weights.clone(), config.n_embd);

        let wpe = embedding(config.n_positions, config.n_embd, vb.pp("wpe"))?;

        let h: Vec<_> = (0..config.n_layer)
            .map(|i| Block::load(vb.pp(&format!("h.{i}")), config))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = LayerNorm::new(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_f"))?;
        let lm_head = Linear::new(wte_weights, None);
        Ok(Self { wte, wpe, h, ln_f, lm_head })
    }
}

impl Block {

    fn forward(&self, x: &Tensor, kv_cache: &mut KVCache, mask_cache: Option<&dyn MaskCache>, layer_idx: usize) -> Result<Tensor> {
        //let (_batch_size, _seq_length, _) = x.dims3()?;

        let residual = x;
        let x = self.ln_1.forward(x)?;


        let attn_output = self.attn.forward(&x, kv_cache, mask_cache, layer_idx)?;

        let x = (attn_output.clone() + residual)?;
        let residual = x.clone();
        let x = self.ln_2.forward(&x)?;

        let ff_output = self.mlp.forward(&x)?;
        let x = (ff_output + residual)?;

        Ok(x)
    }


    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let ln_1 = LayerNorm::new(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = Attention::load(vb.pp("attn"), config)?;
        let ln_2 = LayerNorm::new(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = MLP::load(vb.pp("mlp"), config)?;
        Ok(Self { ln_1, attn, ln_2, mlp })
    }
}

impl Attention {

    fn forward(&self, x: &Tensor, kv_cache: &mut KVCache, mask_cache: Option<&dyn MaskCache>, layer_idx: usize) -> Result<Tensor> {
        let (batch_size, seq_length, _) = x.dims3()?;

        // Apply c_attn Conv1d
        let x = self.c_attn.forward(&x)?;

        // Split into query, key, value
        let chunks = x.chunk(3, D::Minus1)?;
        let query = chunks[0].clone();
        let key = chunks[1].clone();
        let value = chunks[2].clone();

        // Reshape and transpose for multi-head attention
        let query = query.reshape((batch_size, seq_length, self.n_head, self.n_embd / self.n_head))?
            .transpose(1, 2)?;
        let key = key.reshape((batch_size, seq_length, self.n_head, self.n_embd / self.n_head))?
            .transpose(1, 2)?;
        let value = value.reshape((batch_size, seq_length, self.n_head, self.n_embd / self.n_head))?
            .transpose(1, 2)?;

        let (key, value) = if kv_cache.is_enabled() {
            if let Some((past_key, past_value)) = kv_cache.get_layer_cache(layer_idx) {
                // If we have past key and value, concatenate with the new ones
                let key = Tensor::cat(&[past_key, &key], 2)?;
                let value = Tensor::cat(&[past_value, &value], 2)?;
                (key, value)
            } else {
                (key, value)
            }
        } else {
            (key, value)
        };

        if kv_cache.is_enabled() {
            kv_cache.set_layer_cache(layer_idx, (key.clone(), value.clone()));
        }

        let attn_output = self.attention(&query, &key, &value, mask_cache)?;
        let attn_output = attn_output.transpose(1, 2)?
            .reshape((batch_size, seq_length, self.n_embd))?;

        // Apply c_proj Conv1d
        let attn_output = self.c_proj.forward(&attn_output)?;

        Ok(attn_output)
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask_cache: Option<&dyn MaskCache>) -> Result<Tensor> {

        let scale = (self.n_embd as f64 / self.n_head as f64).sqrt();
        let scale_tensor = Tensor::new(scale as f32, query.device())?.to_dtype(query.dtype())?;
        let query = query.broadcast_div(&scale_tensor)?;
        let scores = query.matmul(&key.transpose(2, 3)?)?;
        let (batch_size, num_heads, query_len, key_len) = scores.dims4()?;

        // Retrieve mask from cache
        let causal_mask = Self::get_causal_mask(mask_cache, key_len, batch_size, num_heads, scores.device())?;


        // Adjust the causal mask for the current query length
        let causal_mask = causal_mask.narrow(2, key_len - query_len, query_len)?;

        // Create a tensor of -inf with the same shape as scores
        let mask_value = f32::NEG_INFINITY;
        let query_dtype = query.dtype();

        let neg_inf_tensor = Tensor::full(mask_value, scores.shape(), scores.device())?;
        let neg_inf_tensor = neg_inf_tensor.to_dtype(query_dtype)?;

        // Apply the causal mask
        let masked_scores = causal_mask.where_cond(&scores, &neg_inf_tensor)?;

        let attn_weights = candle_nn::ops::softmax(&masked_scores, D::Minus1)?;

        let attention_out = attn_weights.matmul(value)?;

        Ok(attention_out)
    }


    //HashMap or Vector Version
    fn get_causal_mask(
        mask_cache: Option<&dyn MaskCache>,
        key_len: usize,
        batch_size: usize,
        num_heads: usize,
        device: &Device
    ) -> Result<Tensor> {

        match mask_cache {
            Some(cache) => {
                let mask = cache.get_mask(key_len, batch_size)?;
                Ok(mask)
            },
            None => {
                let mask = Tensor::tril2(key_len, DType::U8, device)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .expand((batch_size, num_heads, key_len, key_len))?;
                Ok(mask)
            }
        }
    }



    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let n_embd = config.n_embd;

        // Load weights for c_attn
        let c_attn_weight = vb.pp("c_attn").get((n_embd, 3 * n_embd), "weight")?;
        // Reshape to [3 * n_embd, n_embd, 1]
        let c_attn_weight = c_attn_weight.transpose(0, 1)?;

        // Load bias for c_attn
        let c_attn_bias = vb.pp("c_attn").get(3 * n_embd, "bias")?;

        // Create c_attn Conv1D layer
        let c_attn = Linear::new(c_attn_weight, Some(c_attn_bias));

        // Load weights for c_proj
        let c_proj_weight = vb.pp("c_proj").get((n_embd, n_embd), "weight")?;
        // Reshape to [n_embd, n_embd, 1]
        let c_proj_weight = c_proj_weight.transpose(0, 1)?;

        // Load bias for c_proj
        let c_proj_bias = vb.pp("c_proj").get(n_embd, "bias")?;

        // Create c_proj Conv1D layer
        let c_proj = Linear::new(c_proj_weight, Some(c_proj_bias));

        Ok(Self {
            c_attn,
            c_proj,
            n_head: config.n_head,
            n_embd,
        })

    }


}



impl MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {

        let h = self.c_fc.forward(&x)?;

        let h = candle_nn::Activation::Gelu.forward(&h)?;

        let h = self.c_proj.forward(&h)?;

        Ok(h)

    }

    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {

        // Load weights for c_fc
        let c_fc_weight = vb.pp("c_fc").get((config.n_embd, 4 * config.n_embd), "weight")?;
        // Reshape to [4 * n_embd, n_embd, 1]
        let c_fc_weight = c_fc_weight.transpose(0, 1)?;

        // Load bias for c_fc
        let c_fc_bias = vb.pp("c_fc").get(4 * config.n_embd, "bias")?;

        // Create c_fc Conv1D layer
        let c_fc = Linear::new(c_fc_weight, Some(c_fc_bias));

        // Load weights for c_proj
        let c_proj_weight = vb.pp("c_proj").get((4 * config.n_embd, config.n_embd), "weight")?;

        // Reshape to [n_embd, 4 * n_embd, 1]
        let c_proj_weight = c_proj_weight.transpose(0, 1)?;

        // Load bias for c_proj
        let c_proj_bias = vb.pp("c_proj").get(config.n_embd, "bias")?;

        // Create c_proj Conv1D layer
        let c_proj = Linear::new(c_proj_weight, Some(c_proj_bias));
        Ok(Self { c_fc, c_proj })
    }
}

#[derive(Debug, Clone)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let x = x.to_dtype(internal_dtype)?;
        let x_shape = x.shape();//.to_vec();
        let mean = x.mean_keepdim(D::Minus1)?;
        let x = (&x - &mean.broadcast_as(x_shape)?)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let eps_tensor = Tensor::new(self.eps, x.device())?.to_dtype(internal_dtype)?.expand(var.shape())?;


        let x = (&x / &((var + eps_tensor)?.sqrt()?).broadcast_as(x_shape)?)?;
        // Broadcast weight and bias to match x's shape

        let weight_broadcast = self.weight.broadcast_as(x_shape)?;
        let bias_broadcast = self.bias.broadcast_as(x_shape)?;
        // Perform the multiplication and addition with broadcasted tensors

        //added to convert back
        let x = x.to_dtype(x_dtype)?;

        let x = (&x * &weight_broadcast + &bias_broadcast)?;

        Ok(x)
    }
}

