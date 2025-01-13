use std::cell::RefCell;
use candid::Principal;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager, VirtualMemory},
    DefaultMemoryImpl, StableBTreeMap,
};
use crate::auth::ensure_authorized;

pub mod auth;
pub mod storage;
pub mod llm;
//pub mod benchmarks; //demo benchmarks

// Re-export common items from storage
pub use storage::{
    Storage,
    CONFIG, SAFETENSORS, TOKENIZER,
    append_config_bytes, config_bytes_length, clear_config_bytes,
    append_safetensors_bytes, safetensors_bytes_length, clear_safetensors_bytes,
    append_tokenizer_bytes, tokenizer_bytes_length, clear_tokenizer_bytes,
};

// Re-export common items from llm
pub use llm::{
    candle::EmptyResult,
    gpt2::Config,
    mask_cache::VecMaskCache,
};


type Memory = VirtualMemory<DefaultMemoryImpl>;

thread_local! {
    pub static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    pub static MAP: RefCell<StableBTreeMap<u8, Vec<u8>, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1))),
        )
    );

    static AUTHORIZED_PRINCIPALS: RefCell<StableBTreeMap<Principal, bool, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(2)))
        )
    );
}

#[ic_cdk::init]
fn init() {
    // Initialize the WASI memory
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);

    // Initialize the application memory (StableBTreeMap)
    let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    MAP.with(|map| {
        *map.borrow_mut() = StableBTreeMap::init(app_memory);
    });

    // Initialize the authorized principals map
    AUTHORIZED_PRINCIPALS.with(|principals| {
        *principals.borrow_mut() = StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(2)))
        );
    });

    let caller = ic_cdk::caller();
    ensure_authorized(caller);
}

#[ic_cdk::pre_upgrade]
fn pre_upgrade() {
    // Save any necessary state before upgrade if needed
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    // Reinitialize the WASI memory after upgrade
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);

    // Reinitialize the application memory (StableBTreeMap) after upgrade
    let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    MAP.with(|map| {
        *map.borrow_mut() = StableBTreeMap::init(app_memory);
    });

    AUTHORIZED_PRINCIPALS.with(|principals| {
        *principals.borrow_mut() = StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(2)))
        );
    });

    let caller = ic_cdk::caller();
    ensure_authorized(caller);
}

