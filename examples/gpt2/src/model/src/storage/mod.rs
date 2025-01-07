// src/storage/mod.rs
mod base;
mod instances;

pub use base::{Storage, GenericStorage, StableStorage};
pub use instances::{
    CONFIG, SAFETENSORS, TOKENIZER,
    append_config_bytes, config_bytes_length, clear_config_bytes,
    append_safetensors_bytes, safetensors_bytes_length, clear_safetensors_bytes,
    append_tokenizer_bytes, tokenizer_bytes_length, clear_tokenizer_bytes,
    store_tokenizer_bytes_to_stable, load_tokenizer_bytes_from_stable,
    store_safetensors_bytes_to_stable, load_safetensors_bytes_from_stable,
};

pub fn call_config_bytes() -> Result<bytes::Bytes, String> {
    CONFIG.call_bytes()
}

pub fn call_safetensors_bytes() -> Result<bytes::Bytes, String> {
    SAFETENSORS.call_bytes()
}

pub fn call_tokenizer_bytes() -> Result<bytes::Bytes, String> {
    TOKENIZER.call_bytes()
}

