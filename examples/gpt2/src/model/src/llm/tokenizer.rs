use std::cell::RefCell;
use anyhow::{Result, anyhow};
use candid::{CandidType, Deserialize};
use tokenizers::Tokenizer;
use crate::auth::is_authenticated;
use crate::storage::call_tokenizer_bytes;

thread_local! {
    static TOKENIZER: RefCell<Option<Tokenizer>> = RefCell::new(None);
}

#[derive(CandidType, Deserialize)]
pub struct TokenizerEncoding {
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
}

#[derive(CandidType, Deserialize)]
pub enum TokenizerResult {
    Ok(TokenizerEncoding),
    Err(String),
}

#[derive(CandidType, Deserialize)]
pub enum DecodingResult {
    Ok(Vec<String>),
    Err(String),
}

fn setup() -> Result<()> {
    let bytes = call_tokenizer_bytes()
        .map_err(|e| anyhow!("Failed to get tokenizer bytes: {}", e))?;

    let tokenizer = Tokenizer::from_bytes(bytes)
        .map_err(|e| anyhow!("Failed to create tokenizer: {}", e))?;

    TOKENIZER.with(|t| {
        *t.borrow_mut() = Some(tokenizer);
    });

    Ok(())
}

#[ic_cdk::update(guard = "is_authenticated")]
fn setup_tokenizer() -> Result<(), String> {
    setup().map_err(|err| format!("Failed to setup tokenizer: {}", err))
}

#[ic_cdk::query]
pub fn tokenize(input_text: String) -> TokenizerResult {
    TOKENIZER.with(|t| {
        let tokenizer = t.borrow();
        let tokenizer = match tokenizer.as_ref() {
            Some(t) => t,
            None => return TokenizerResult::Err("Tokenizer not initialized".to_string()),
        };

        match tokenizer.encode(input_text, true) {
            Ok(encoding) => {
                let token_ids = encoding.get_ids().to_vec();
                let tokens = encoding.get_tokens().to_vec();
                TokenizerResult::Ok(TokenizerEncoding { tokens, token_ids })
            },
            Err(e) => TokenizerResult::Err(format!("Failed to encode text: {}", e)),
        }
    })
}

#[ic_cdk::query]
fn decode(input_token_ids: Vec<u32>) -> DecodingResult {
    TOKENIZER.with(|t| {
        let tokenizer = t.borrow();
        match tokenizer.as_ref() {
            None => DecodingResult::Err("Tokenizer not initialized".to_string()),
            Some(tokenizer) => {
                // Decode each token individually
                let mut token_strings = Vec::new();
                for &token_id in &input_token_ids {
                    match tokenizer.decode(&[token_id], true) {
                        Ok(decoded_text) => token_strings.push(decoded_text),
                        Err(e) => return DecodingResult::Err(format!("Failed to decode token {}: {}", token_id, e)),
                    }
                }
                DecodingResult::Ok(token_strings)
            }
        }
    })
}

#[ic_cdk::query]
pub fn decode_batch(input_token_ids: Vec<u32>) -> Result<String, String> {
    TOKENIZER.with(|t| {
        let tokenizer = t.borrow();
        match tokenizer.as_ref() {
            None => Err("Tokenizer not initialized".to_string()),
            Some(tokenizer) => {
                match tokenizer.decode(&input_token_ids, true) {
                    Ok(decoded_text) => Ok(decoded_text),
                    Err(e) => Err(format!("Failed to decode tokens: {}", e))
                }
            }
        }
    })
}