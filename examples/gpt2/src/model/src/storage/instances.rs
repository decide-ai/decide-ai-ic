// storage/instances.rs
use super::declare_storage;
use crate::auth::is_authenticated;
use crate::storage::GenericStorage;
//use std::cell::RefCell;
use crate::storage::base::StableStorage;
use crate::storage::base::Storage;

declare_storage! {
    /// Configuration storage
    pub CONFIG {
        name: "config",
        stable_key: 2
    }
}

declare_storage! {
    /// SafeTensors model weights storage
    pub SAFETENSORS {
        name: "safetensors",
        stable_key: 0
    }
}

declare_storage! {
    /// Tokenizer data storage
    pub TOKENIZER {
        name: "tokenizer",
        stable_key: 1
    }
}