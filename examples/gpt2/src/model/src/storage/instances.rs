// storage/instances.rs
use std::sync::RwLock;
use crate::auth::is_authenticated;
use crate::storage::GenericStorage;
use crate::storage::base::{Storage, StableStorage};

static CONFIG_CELL: RwLock<Vec<u8>> = RwLock::new(vec![]);
static SAFETENSORS_CELL: RwLock<Vec<u8>> = RwLock::new(vec![]);
static TOKENIZER_CELL: RwLock<Vec<u8>> = RwLock::new(vec![]);

pub static CONFIG: GenericStorage = GenericStorage::new("config", &CONFIG_CELL);
pub static SAFETENSORS: GenericStorage = GenericStorage::new("safetensors", &SAFETENSORS_CELL);
pub static TOKENIZER: GenericStorage = GenericStorage::new("tokenizer", &TOKENIZER_CELL);

// IC interface functions
#[ic_cdk::update(guard = "is_authenticated")]
pub fn append_config_bytes(bytes: Vec<u8>) {
    CONFIG.append_bytes(bytes);
}

#[ic_cdk::query]
pub fn config_bytes_length() -> usize {
    CONFIG.bytes_length()
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn clear_config_bytes() {
    CONFIG.clear_bytes();
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn append_safetensors_bytes(bytes: Vec<u8>) {
    SAFETENSORS.append_bytes(bytes);
}

#[ic_cdk::query]
pub fn safetensors_bytes_length() -> usize {
    SAFETENSORS.bytes_length()
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn clear_safetensors_bytes() {
    SAFETENSORS.clear_bytes();
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn append_tokenizer_bytes(bytes: Vec<u8>) {
    TOKENIZER.append_bytes(bytes);
}

#[ic_cdk::query]
pub fn tokenizer_bytes_length() -> usize {
    TOKENIZER.bytes_length()
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn clear_tokenizer_bytes() {
    TOKENIZER.clear_bytes();
}


#[ic_cdk::update(guard = "is_authenticated")]
pub fn store_tokenizer_bytes_to_stable() -> Result<(), String> {
    TOKENIZER.store_to_stable(1)
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn load_tokenizer_bytes_from_stable() -> Result<(), String> {
    TOKENIZER.load_from_stable(1)
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn store_safetensors_bytes_to_stable() -> Result<(), String> {
    SAFETENSORS.store_to_stable(0)
}

#[ic_cdk::update(guard = "is_authenticated")]
pub fn load_safetensors_bytes_from_stable() -> Result<(), String> {
    SAFETENSORS.load_from_stable(0)
}