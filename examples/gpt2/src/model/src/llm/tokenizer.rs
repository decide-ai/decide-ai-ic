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
pub fn setup_tokenizer() -> Result<(), String> {
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










#[cfg(feature = "canbench-rs")]
mod tokenizer_benchmarks {
    use super::*;
    use canbench_rs::bench;

    const SMALL_TEST_INPUT: &str = "Hello world";

    // Benchmark just the RefCell borrow operation
    #[bench]
    fn benchmark_tokenizer_borrow() {
        TOKENIZER.with(|t| {
            let _tokenizer = t.borrow();
        });
    }

    // Benchmark RefCell borrow and tokenizer access
    #[bench]
    fn benchmark_tokenizer_access() {
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            let _tokenizer = match tokenizer.as_ref() {
                Some(t) => t,
                None => return,
            };
        });
    }

    // Benchmark just the encode operation with small input
    #[bench]
    fn benchmark_encode_small() {
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            if let Some(t) = tokenizer.as_ref() {
                let _ = t.encode(SMALL_TEST_INPUT.to_string(), true);
            }
        });
    }

    // Benchmark the vector conversions
    #[bench]
    fn benchmark_vector_conversion() {
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            if let Some(t) = tokenizer.as_ref() {
                if let Ok(encoding) = t.encode(SMALL_TEST_INPUT.to_string(), true) {
                    let _token_ids = encoding.get_ids().to_vec();
                    let _tokens = encoding.get_tokens().to_vec();
                }
            }
        });
    }

    // Benchmark error case handling (with uninitialized tokenizer)
    #[bench]
    fn benchmark_error_handling() {
        // Note: This assumes you have a way to temporarily clear the tokenizer
        // You might need to modify this based on your actual implementation
        TOKENIZER.with(|t| {
            let tokenizer = t.borrow();
            if tokenizer.as_ref().is_none() {
                let _ = TokenizerResult::Err("Tokenizer not initialized".to_string());
            }
        });
    }

    // Original benchmarks for comparison
    #[bench]
    fn benchmark_tokenizer_small_full() {
        let _ = tokenize(SMALL_TEST_INPUT.to_string());
    }

    #[bench]
    pub fn benchmark_tokenizer_small() {
        const SMALL_TEST_INPUT: &str = "Hello world"; // Smaller input for testing

        let _token_ids = match tokenize(SMALL_TEST_INPUT.to_string()) {
            TokenizerResult::Ok(encoding) => Ok(encoding.token_ids),
            TokenizerResult::Err(e) => Err(format!("Tokenization failed: {}", e)),
        };

    }


    #[bench]
    pub fn benchmark_tokenizer_large() {

        const LARGE_TEST_INPUT: &str =
"Artificial intelligence (AI) is revolutionizing industries worldwide. \
From healthcare to finance, transportation to entertainment, AI's applications \
are vast and rapidly growing. Imagine a world where autonomous vehicles navigate \
bustling city streets, personalized medicine tailors treatments to individual \
patients, and advanced analytics predict market trends with unparalleled accuracy. \
At the heart of AI are algorithms that learn, adapt, and evolve—ushering in an \
era of unprecedented innovation.

The journey of AI began decades ago, with early research into machine learning and natural language processing. \
Today, these technologies power chatbots that can simulate human-like conversations, \
recommendation systems that suggest movies and products you’ll love, \
and even AI artists creating unique pieces of digital artwork. However, with great power comes great responsibility. \
Ethical considerations, data privacy, and transparency are critical issues that developers and policymakers must address as AI becomes an integral part of daily life.

As we look to the future, one question looms large: How will humanity coexist with increasingly intelligent systems? \
Collaboration between humans and AI could unlock new possibilities, \
from solving complex scientific challenges to improving the quality of life for people around the globe. \
Yet, careful oversight is essential to ensure these technologies serve the common good, \
rather than exacerbating inequality or harm.

Embracing AI's potential while mitigating its risks will require a collaborative effort from researchers, \
businesses, and governments alike. By fostering an open dialogue about AI's benefits and challenges, \
we can chart a course toward a future where artificial intelligence complements and enhances human capabilities, \
rather than replacing them. The time to act is now—together, we can shape the future of AI and its role in society.";

        let _token_ids = match tokenize(LARGE_TEST_INPUT.to_string()) {
            TokenizerResult::Ok(encoding) => Ok(encoding.token_ids),
            TokenizerResult::Err(e) => Err(format!("Tokenization failed: {}", e)),
        };
    }

}