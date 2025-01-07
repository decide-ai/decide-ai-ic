// src/storage/base.rs
use std::cell::RefCell;
use bytes::Bytes;
use crate::auth::is_authenticated;
use crate::MAP;
use std::sync::RwLock;
use paste::paste;

// Core storage trait
pub trait Storage {
    fn append_bytes(&self, bytes: Vec<u8>);
    fn bytes_length(&self) -> usize;
    fn clear_bytes(&self);
    fn call_bytes(&self) -> Result<Bytes, String>;
}


pub struct GenericStorage {
    name: &'static str,
    ref_cell: &'static RwLock<Vec<u8>>,
}

impl GenericStorage {
    pub const fn new(name: &'static str, ref_cell: &'static RwLock<Vec<u8>>) -> Self {
        Self { name, ref_cell }
    }
}

impl Storage for GenericStorage {
    fn append_bytes(&self, bytes: Vec<u8>) {
        self.ref_cell.write().unwrap().extend(bytes);
    }

    fn bytes_length(&self) -> usize {
        self.ref_cell.read().unwrap().len()
    }

    fn clear_bytes(&self) {
        self.ref_cell.write().unwrap().clear();
    }

    fn call_bytes(&self) -> Result<Bytes, String> {
        let mut data = Vec::new();
        std::mem::swap(&mut data, &mut *self.ref_cell.write().unwrap());
        Ok(Bytes::from(data))
    }
}

// Helper struct for IC methods
pub struct StorageIcMethods {
    pub append_bytes: Box<dyn Fn(Vec<u8>)>,
    pub bytes_length: Box<dyn Fn() -> usize>,
    pub clear_bytes: Box<dyn Fn()>,
}

// Stable storage trait
pub trait StableStorage {
    fn store_to_stable(&self, key: u8) -> Result<(), String>;
    fn load_from_stable(&self, key: u8) -> Result<(), String>;
}

impl StableStorage for GenericStorage {
    fn store_to_stable(&self, key: u8) -> Result<(), String> {
        let bytes = self.call_bytes()
            .map_err(|e| format!("Failed to get {} bytes: {}", self.name, e))?;

        MAP.with(|p| {
            let mut map = p.borrow_mut();
            map.insert(key, bytes.to_vec());
        });

        Ok(())
    }

    fn load_from_stable(&self, key: u8) -> Result<(), String> {
        MAP.with(|p| {
            if let Some(data) = p.borrow().get(&key) {
                self.clear_bytes();
                self.append_bytes(data.clone());
                Ok(())
            } else {
                Err(format!("No {} data found in stable storage", self.name))
            }
        })
    }
}

// Macros
#[macro_export]
macro_rules! create_storage {
    ($vis:vis $name:ident, $storage_name:expr) => {
        thread_local! {
            static REF_CELL: RefCell<Vec<u8>> = RefCell::new(vec![]);
        }

        $vis static $name: GenericStorage = GenericStorage::new($storage_name, &REF_CELL);

        #[ic_cdk::update(guard = "is_authenticated")]
        $vis fn paste_ident!(append_, $name:lower, _bytes)(bytes: Vec<u8>) {
            $name.append_bytes(bytes);
        }

        #[ic_cdk::query]
        $vis fn paste_ident!($name:lower, _bytes_length)() -> usize {
            $name.bytes_length()
        }

        #[ic_cdk::update(guard = "is_authenticated")]
        $vis fn paste_ident!(clear_, $name:lower, _bytes)() {
            $name.clear_bytes();
        }
    };
}

#[macro_export]
macro_rules! create_stable_storage_methods {
    ($storage:expr, $key:expr) => {
        #[ic_cdk::update(guard = "is_authenticated")]
        pub fn paste_ident!(store_, $storage:lower, _bytes_to_stable)() -> Result<(), String> {
            $storage.store_to_stable($key)
        }

        #[ic_cdk::update(guard = "is_authenticated")]
        pub fn paste_ident!(load_, $storage:lower, _bytes_from_stable)() -> Result<(), String> {
            $storage.load_from_stable($key)
        }
    };
}