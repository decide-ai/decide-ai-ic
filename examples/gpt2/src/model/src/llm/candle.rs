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
        //let config_bytes = include_bytes!("./canbench_assets/config.json");
        let config_bytes = include_bytes!("../../../../canbench_assets/config.json");
        let config: Config = from_slice(config_bytes)
            .map_err(|e| {
                println!("Failed to parse config: {}", e);
                anyhow!("Config parse error: {}", e)
            })?;

        // Calculate required size
        const MODEL_SIZE: usize = 510368814; // Your exact file size
        let pages = ic_cdk::api::stable::stable_size();
        let available_bytes = pages as usize * 65536;

        println!("Stable memory pages: {}, bytes available: {}", pages, available_bytes);

        if available_bytes < MODEL_SIZE {
            return Err(anyhow!("Not enough stable memory. Need {} bytes, have {}",
                MODEL_SIZE, available_bytes));
        }

        // Read the model
        let mut model_bytes = vec![0u8; MODEL_SIZE];
        ic_cdk::api::stable::stable_read(0, &mut model_bytes);
        println!("Read {} bytes from stable memory", MODEL_SIZE);

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
    fn initialization_only() -> canbench_rs::BenchResult {
        // Clear any existing model state
        GPT2_MODEL.with(|cell| {
            *cell.borrow_mut() = None;
        });
        GPT2_KV_CACHE.with(|cell| {
            *cell.borrow_mut() = None;
        });
        GPT2_MASK_CACHE.with(|cell| {
            *cell.borrow_mut() = None;
        });

        canbench_rs::bench_fn(|| {
            // Do the actual initialization inside the benchmark
            if let Err(e) = initialize_model() {
                println!("Failed to initialize model: {}", e);
                return;
            }

            // Verify initialization
            let model_state = GPT2_MODEL.with(|cell| {
                let is_some = cell.borrow().is_some();
                println!("Model loaded state: {}", is_some);
                is_some
            });

            if !model_state {
                println!("Model not properly initialized");
            }
        })
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



    use paste::paste;

    fn create_prompt(length: usize) -> Vec<u32> {
        let mut prompt = Vec::with_capacity(length);
        for i in 0..length {
            prompt.push(BASE_PROMPT[i % BASE_PROMPT.len()]);
        }
        prompt
    }

    const BASE_PROMPT: &[u32] = &[1, 2, 3, 4];


    // Define individual benchmark functions instead of using nested repetition
    macro_rules! define_bench_fn {
        ($input_len:expr, $gen_len:expr) => {
            paste! {
                #[bench(raw)]
                fn [<inference_bench_input_ $input_len _gen_ $gen_len>]() -> canbench_rs::BenchResult {
                    println!("Starting inference benchmark with input length {} and gen length {}",
                            $input_len, $gen_len);

                    // Initialize model if needed
                    let model_state = GPT2_MODEL.with(|cell| cell.borrow().is_some());
                    if !model_state {
                        if let Err(e) = initialize_model() {
                            println!("Failed to initialize model: {}", e);
                            return canbench_rs::bench_fn(|| {});
                        }
                    }

                    let prompt = create_prompt($input_len);

                    canbench_rs::bench_fn(|| {
                        match internal_inference(
                            prompt.clone(),
                            $gen_len,
                            TYPICAL_TEMP,
                            EOS_TOKEN
                        ) {
                            Ok(tokens) => println!(
                                "Input len: {}, Gen len: {}, Output tokens: {}",
                                $input_len, $gen_len, tokens.len()
                            ),
                            Err(e) => println!("Inference failed: {}", e),
                        };
                    })
                }
            }
        };
    }

    // Call the macro for each combination explicitly
    define_bench_fn!(1, 1);
    define_bench_fn!(1, 2);
    define_bench_fn!(1, 4);
    define_bench_fn!(1, 8);
    define_bench_fn!(2, 1);
    define_bench_fn!(2, 2);
    define_bench_fn!(2, 4);
    define_bench_fn!(2, 8);
    define_bench_fn!(4, 1);
    define_bench_fn!(4, 2);
    define_bench_fn!(4, 4);
    define_bench_fn!(4, 8);
    define_bench_fn!(8, 1);
    define_bench_fn!(8, 2);
    define_bench_fn!(8, 4);
    define_bench_fn!(8, 8);
    define_bench_fn!(16, 1);
    define_bench_fn!(16, 2);
    define_bench_fn!(16, 4);
    define_bench_fn!(16, 8);
    define_bench_fn!(32, 1);
    define_bench_fn!(32, 2);
    define_bench_fn!(32, 4);
    define_bench_fn!(32, 8);
    define_bench_fn!(64, 1);
    define_bench_fn!(64, 2);
    define_bench_fn!(64, 4);
    define_bench_fn!(64, 8);
    define_bench_fn!(128, 1);
    define_bench_fn!(128, 2);
    define_bench_fn!(128, 4);
    define_bench_fn!(128, 8);
    define_bench_fn!(256, 1);
    define_bench_fn!(256, 2);
    define_bench_fn!(256, 4);
    define_bench_fn!(256, 8);
    define_bench_fn!(512, 1);
    define_bench_fn!(512, 2);
    define_bench_fn!(512, 4);
    define_bench_fn!(512, 8);
    define_bench_fn!(1024, 1);
    define_bench_fn!(1024, 2);
    define_bench_fn!(1024, 4);
    define_bench_fn!(1024, 8);
}