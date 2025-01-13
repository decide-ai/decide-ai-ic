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




#[cfg(feature = "canbench-rs")]
mod inference_benchmarks {
    use super::*;
    use canbench_rs::bench;
    use std::println; // Add explicit println import

    const TYPICAL_PROMPT: [u32; 4] = [1, 2, 3, 4];
    const TYPICAL_TEMP: f64 = 0.7;
    const EOS_TOKEN: u32 = 50257;

    fn initialize_model() -> Result<(), anyhow::Error> {
        println!("Starting model initialization...");

        let device = Device::Cpu;
        let dtype = DType::F32;

        // Load and verify config
        println!("Loading config file...");
        let config_bytes = match std::fs::read("canbench_assets/config.json") {
            Ok(bytes) => {
                println!("Config loaded successfully, size: {} bytes", bytes.len());
                bytes
            },
            Err(e) => {
                println!("Failed to load config: {}", e);
                return Err(anyhow!("Config load error: {}", e));
            }
        };

        let config: Config = from_slice(&config_bytes)
            .map_err(|e| {
                println!("Failed to parse config: {}", e);
                anyhow!("Config parse error: {}", e)
            })?;

        // Load and verify model
        println!("Loading model file...");
        let model_bytes = match std::fs::read("canbench_assets/model.safetensors") {
            Ok(bytes) => {
                println!("Model loaded successfully, size: {} bytes", bytes.len());
                bytes
            },
            Err(e) => {
                println!("Failed to load model: {}", e);
                return Err(anyhow!("Model load error: {}", e));
            }
        };

        println!("Creating VarBuilder...");
        let vb = VarBuilder::from_slice_safetensors(&model_bytes, dtype, &device)?;

        // Initialize caches
        println!("Initializing caches...");
        GPT2_KV_CACHE.with(|cell| {
            let cache = KVCache::new(config.n_layer, true);
            *cell.borrow_mut() = Some(cache);
        });

        GPT2_MASK_CACHE.with(|cell| {
            let mask_cache = VecMaskCache::new(107, config.n_head, device.clone())
                .expect("Failed to create VecMaskCache");
            *cell.borrow_mut() = Some(mask_cache);
        });

        println!("Loading GPT2 model...");
        GPT2_MODEL.with(|cell| -> Result<(), anyhow::Error> {
            let model = GPT2::load(vb.pp("transformer"), &config)?;
            *cell.borrow_mut() = Some(model);
            println!("Model loaded successfully!");
            Ok(())
        })?;

        Ok(())
    }

    #[bench(raw)]
    fn inference_bench() -> canbench_rs::BenchResult {
        println!("Starting inference benchmark...");

        // Initialize model state
        match initialize_model() {
            Ok(_) => println!("Model initialized successfully"),
            Err(e) => {
                println!("Failed to initialize model: {}", e);
                return canbench_rs::bench_fn(|| {});
            }
        }

        let model_state = GPT2_MODEL.with(|cell| {
            let is_some = cell.borrow().is_some();
            println!("Model loaded state: {}", is_some);
            is_some
        });

        if !model_state {
            println!("Model not properly initialized");
            return canbench_rs::bench_fn(|| {});
        }

        println!("Starting inference with prompt length: {}", TYPICAL_PROMPT.len());

        canbench_rs::bench_fn(|| {
            match internal_inference(
                TYPICAL_PROMPT.to_vec(),
                5,
                TYPICAL_TEMP,
                EOS_TOKEN
            ) {
                Ok(tokens) => println!("Inference generated {} tokens", tokens.len()),
                Err(e) => println!("Inference failed: {}", e),
            };
        })
    }

    #[bench(raw)]
    fn model_state_check() -> canbench_rs::BenchResult {
        println!("Running model state check...");
        canbench_rs::bench_fn(|| {
            let model_init = GPT2_MODEL.with(|cell| cell.borrow().is_some());
            let cache_init = GPT2_KV_CACHE.with(|cell| cell.borrow().is_some());
            let mask_init = GPT2_MASK_CACHE.with(|cell| cell.borrow().is_some());

            println!(
                "Model State - Model: {}, Cache: {}, Mask: {}",
                model_init, cache_init, mask_init
            );
        })
    }
}