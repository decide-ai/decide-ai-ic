//src/llm/mod.rs

pub mod mask_cache;
pub mod gpt2;
pub mod candle;
pub mod sample;
pub mod tokenizer;

// Re-export common items
pub use candle::EmptyResult;
pub use gpt2::Config;
pub use mask_cache::VecMaskCache;

//use crate::llm::candle::{TokenIDsResult, InferenceResult, InferenceRecord, internal_inference};
use crate::llm::candle::{internal_inference};
use crate::llm::tokenizer::{TokenizerResult, tokenize, decode_batch};
//use crate::auth::is_authenticated;

const MAX_TOKENS: u8 = 100;
const MIN_TEMP: f64 = 0.0;
const MAX_TEMP: f64 = 2.0;


const USER_TOKEN: u32 = 50258;
const NEWLINE_TOKEN: u32 = 198;
const END_USER_RESPONSE_TOKEN: u32 = 628;
const ASSISTANT_TOKEN: u32 = 50259;


#[ic_cdk::update]
fn generate(input_text: String, gen_iter: u8, temperature: f64) -> Result<String, String> {

    if gen_iter > MAX_TOKENS {
        return Err(format!("Token count exceeds maximum limit of {}", MAX_TOKENS));
    }
    if temperature < MIN_TEMP || temperature > MAX_TEMP {
        return Err(format!("Temperature must be between {} and {}", MIN_TEMP, MAX_TEMP));
    }

    // First tokenize the input
    let tokens = match tokenize(input_text) {
        TokenizerResult::Ok(encoding) => encoding.token_ids,
        TokenizerResult::Err(e) => return Err(format!("Tokenization failed: {}", e)),
    };
    let mut input_tokens = vec![USER_TOKEN, NEWLINE_TOKEN];
    input_tokens.extend(tokens);
    input_tokens.extend(vec![END_USER_RESPONSE_TOKEN, ASSISTANT_TOKEN, NEWLINE_TOKEN]);
    ic_cdk::println!("input tokens{:?}",input_tokens);

    // Then run inference with the tokens
    let generated_tokens = match internal_inference(input_tokens, gen_iter, temperature.into(), 50257_u32) {
        Ok(tokens) => tokens,
        Err(e) => return Err(format!("Inference failed: {}", e)),
    };
    ic_cdk::println!("generated tokens{:?}",generated_tokens);

    // Finally decode the generated tokens
    match decode_batch(generated_tokens) {
        Ok(text) => Ok(text),
        Err(e) => Err(format!("Decoding failed: {}", e)),
    }
}