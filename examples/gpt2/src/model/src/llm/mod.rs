pub mod mask_cache;
pub mod gpt2;
pub mod candle;
pub mod sample;
pub mod tokenizer;

// Re-export common items
pub use candle::EmptyResult;
pub use gpt2::Config;
pub use mask_cache::VecMaskCache;

use crate::llm::candle::{TokenIDsResult, InferenceResult, InferenceRecord, internal_inference};
use crate::llm::tokenizer::{TokenizerResult, tokenize, decode_batch};
use crate::auth::is_authenticated;



#[ic_cdk::update(guard = "is_authenticated")]
fn generate(input_text: String, gen_iter: u8, temperature: f64) -> Result<String, String> {
    // First tokenize the input
    let tokens = match tokenize(input_text) {
        TokenizerResult::Ok(encoding) => encoding.token_ids,
        TokenizerResult::Err(e) => return Err(format!("Tokenization failed: {}", e)),
    };

    // Then run inference with the tokens
    let generated_tokens = match internal_inference(tokens, gen_iter, temperature.into()) {
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