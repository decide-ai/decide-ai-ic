use std::cell::RefCell;
use serde::Deserialize;
use candid::CandidType;
use serde_json::from_slice;
use crate::auth::is_authenticated;
use crate::storage::{
    Storage, CONFIG, SAFETENSORS,
};
use crate::llm::{
    sample,
    //gpt2::{GPT2, Config, KVCache, MaskCache},
    gpt2::{GPT2, Config, KVCache},
    mask_cache::VecMaskCache,
};
use candle_nn::VarBuilder;
use candle::{DType, Tensor, Device};
use anyhow::{anyhow, Result};

thread_local! {
    static GPT2_MODEL: RefCell<Option<GPT2>> = RefCell::new(None);
    static GPT2_KV_CACHE: RefCell<Option<KVCache>> = RefCell::new(None);
    static GPT2_MASK_CACHE: RefCell<Option<VecMaskCache>> = RefCell::new(None);
}

#[derive(CandidType, Deserialize)]
pub enum EmptyResult {
    Ok,
    Err(String),
}


fn internal_setup_model() -> Result<(), anyhow::Error> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let config_bytes = CONFIG.call_bytes()
        .map_err(|e| anyhow!("Failed to get config bytes: {}", e))?;

    let config: Config = from_slice(&config_bytes)
        .map_err(|e| anyhow!("Failed to parse config: {}", e))?;

    let safetensors_bytes = SAFETENSORS.call_bytes()
        .map_err(|e| anyhow!("Failed to get safetensors bytes: {}", e))?;

    let safetensors_slice = safetensors_bytes.as_ref();

    //let vb = unsafe { VarBuilder::from_slice_safetensors(safetensors_slice, dtype, &device)? };
    let vb = VarBuilder::from_slice_safetensors(safetensors_slice, dtype, &device)?;

    GPT2_KV_CACHE.with(|cell| {
        let cache = KVCache::new(config.n_layer, true);  // Enable caching
        *cell.borrow_mut() = Some(cache);
    });

    GPT2_MASK_CACHE.with(|cell| {
        let mask_cache = VecMaskCache::new(107, config.n_head, device.clone())
            .expect("Failed to create VecMaskCache");
        *cell.borrow_mut() = Some(mask_cache);
    });

    GPT2_MODEL.with(|cell| -> Result<(), anyhow::Error> {
        //let model = GPT2::load(vb, &config)?; //standard GPT2
        let model = GPT2::load(vb.pp("transformer"), &config)?; //GPT2-Instruct
        *cell.borrow_mut() = Some(model);
        Ok(())
    })?;

    Ok(())
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn setup_model() -> EmptyResult {
    match internal_setup_model() {
        Ok(_) => EmptyResult::Ok,
        Err(e) => EmptyResult::Err(e.to_string()),
    }
}






#[derive(CandidType, Deserialize)]
pub enum TokenIDsResult {
    Ok(Vec<u32>),
    Err(String),
}

#[derive(CandidType, Deserialize)]
pub struct InferenceRecord {
    pub result: TokenIDsResult,
}

#[derive(CandidType, Deserialize)]
pub enum InferenceResult {
    Ok(InferenceRecord),
    Err(String),
}


#[ic_cdk::update(guard = "is_authenticated")]
fn inference(tokens: Vec<u32>, gen_iter: u8, temperature: f64) -> InferenceResult {
    match internal_inference(tokens, gen_iter, temperature.into(), 50257_u32) {
        Ok(generated_tokens) => {
            InferenceResult::Ok(InferenceRecord {
                result: TokenIDsResult::Ok(generated_tokens),
            })
        },
        Err(e) => {
            InferenceResult::Err(e.to_string())
        },
    }
}



pub fn internal_inference(tokens: Vec<u32>, gen_iter: u8, temperature: f64, eos: u32) -> Result<Vec<u32>, anyhow::Error> {
    let device = Device::Cpu;
    let mut input = Tensor::new(tokens.as_slice(), &device)?
        .reshape((1, tokens.len()))?;
    let mut gen_token_ids = vec![];

    GPT2_MASK_CACHE.with(|mask_cell| {
        GPT2_MODEL.with(|model_cell| {
            GPT2_KV_CACHE.with(|cache_cell| -> Result<Vec<u32>, anyhow::Error> {
                let model = model_cell.borrow();
                let mut cache = cache_cell.borrow_mut();
                let mask_cache = mask_cell.borrow();

                let model = model.as_ref().ok_or_else(|| anyhow!("model not initialized"))?;
                let cache = cache.as_mut().ok_or_else(|| anyhow!("kv-cache not initialized"))?;
                let mask_cache = mask_cache.as_ref().ok_or_else(|| anyhow!("mask cache not initialized"))?;

                // Reset the KV cache at the start of inference
                cache.clear();

                for _ in 0..gen_iter {

                    // Perform forward pass and sampling
                    let logits = model.forward(&input, cache, Some(mask_cache))?;
                    let logits = logits.squeeze(0)?;
                    let last_logits = logits.get(logits.dim(0)? - 1)?;
                    let next_token = sample::sample(&last_logits, temperature, None, None)?;

                    // Add next token to generated tokens
                    gen_token_ids.push(next_token);

                    // Check for EOS and break if reached
                    if eos == next_token {
                        break;
                    }

                    // Update input for the next iteration
                    input = Tensor::new(vec![next_token], &device)?.reshape((1, 1))?;

                }

                Ok(gen_token_ids)
            })
        })
    })
}






