// src/storage/mod.rs
mod base;
mod instances;

pub use base::{Storage, GenericStorage, StableStorage};

// Re-export the macro for use in other modules
pub use crate::declare_storage;

// Re-export the storage instances and their functions
pub use instances::{
    CONFIG, SAFETENSORS, TOKENIZER,
    append_config_bytes, config_bytes_length, clear_config_bytes, call_config_bytes,
    store_config_bytes_to_stable, load_config_bytes_from_stable,

    append_safetensors_bytes, safetensors_bytes_length, clear_safetensors_bytes, call_safetensors_bytes,
    store_safetensors_bytes_to_stable, load_safetensors_bytes_from_stable,

    append_tokenizer_bytes, tokenizer_bytes_length, clear_tokenizer_bytes, call_tokenizer_bytes,
    store_tokenizer_bytes_to_stable, load_tokenizer_bytes_from_stable,
};