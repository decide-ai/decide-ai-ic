// src/model/src/llm/benchmarks.rs
use super::*;

/*
const TEST_INPUT: &str = "What is the meaning of life?";
const TEST_TEMPERATURE: f64 = 0.7;
const TEST_GEN_ITER: u8 = 50;

#[ic_cdk::update]
pub fn benchmark_inference() -> Result<Vec<u32>, String> {
    let tokens = match tokenize(TEST_INPUT.to_string()) {
        TokenizerResult::Ok(encoding) => encoding.token_ids,
        TokenizerResult::Err(e) => return Err(format!("Tokenization failed: {}", e)),
    };
    let mut input_tokens = vec![USER_TOKEN, NEWLINE_TOKEN];
    input_tokens.extend(tokens);
    input_tokens.extend(vec![END_USER_RESPONSE_TOKEN, ASSISTANT_TOKEN, NEWLINE_TOKEN]);

    // Map the anyhow::Error to String
    internal_inference(
        input_tokens,
        TEST_GEN_ITER,
        TEST_TEMPERATURE.into(),
        50257_u32
    ).map_err(|e| e.to_string())
}
*/

const SMALL_TEST_INPUT: &str = "Hello world"; // Smaller input for testing


pub fn benchmark_tokenizer_small() -> Result<Vec<u32>, String> {
    match tokenize(SMALL_TEST_INPUT.to_string()) {
        TokenizerResult::Ok(encoding) => Ok(encoding.token_ids),
        TokenizerResult::Err(e) => Err(format!("Tokenization failed: {}", e)),
    }
}