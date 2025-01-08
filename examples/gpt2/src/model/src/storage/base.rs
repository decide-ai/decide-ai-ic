// src/storage/base.rs
use std::cell::RefCell;
use crate::MAP;

// We know this is safe because IC canisters are single-threaded
pub struct ICRefCell<T>(RefCell<T>);
unsafe impl<T> Sync for ICRefCell<T> {}

impl<T> ICRefCell<T> {
    pub const fn new(value: T) -> Self {
        Self(RefCell::new(value))
    }
}

// Core storage trait
pub trait Storage {
    fn append_bytes(&self, bytes: Vec<u8>);
    fn bytes_length(&self) -> usize;
    fn clear_bytes(&self);
    fn call_bytes(&self) -> Result<Vec<u8>, String>;
}

pub struct GenericStorage {
    name: &'static str,
    cell: &'static ICRefCell<Vec<u8>>,
}

impl GenericStorage {
    pub const fn new(name: &'static str, cell: &'static ICRefCell<Vec<u8>>) -> Self {
        Self { name, cell }
    }
}

impl Storage for GenericStorage {
    fn append_bytes(&self, bytes: Vec<u8>) {
        self.cell.0.borrow_mut().extend(bytes);
    }

    fn bytes_length(&self) -> usize {
        self.cell.0.borrow().len()
    }

    fn clear_bytes(&self) {
        self.cell.0.borrow_mut().clear();
    }

    fn call_bytes(&self) -> Result<Vec<u8>, String> {
        let mut data = Vec::new();
        std::mem::swap(&mut data, &mut *self.cell.0.borrow_mut());
        Ok(data)
    }
}

// Optional stable storage trait
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
            map.insert(key, bytes);
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

// New macro for declaring storage
#[macro_export]
macro_rules! declare_storage {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident {
            name: $storage_name:expr,
            $(stable_key: $stable_key:expr)?
        }
    ) => {
        paste::paste! {
            static [<$name _CELL>]: crate::storage::base::ICRefCell<Vec<u8>> =
                crate::storage::base::ICRefCell::new(Vec::new());

            $(#[$meta])*
            $vis static $name: GenericStorage =
                GenericStorage::new($storage_name, &[<$name _CELL>]);

            #[ic_cdk::update(guard = "is_authenticated")]
            $vis fn [<append_ $name:lower _bytes>](bytes: Vec<u8>) {
                $name.append_bytes(bytes);
            }

            #[ic_cdk::query]
            $vis fn [<$name:lower _bytes_length>]() -> usize {
                $name.bytes_length()
            }

            #[ic_cdk::update(guard = "is_authenticated")]
            $vis fn [<clear_ $name:lower _bytes>]() {
                $name.clear_bytes();
            }

            #[ic_cdk::update(guard = "is_authenticated")]
            $vis fn [<call_ $name:lower _bytes>]() -> Result<Vec<u8>, String> {
                $name.call_bytes()
            }

            $(
                #[ic_cdk::update(guard = "is_authenticated")]
                $vis fn [<store_ $name:lower _bytes_to_stable>]() -> Result<(), String> {
                    $name.store_to_stable($stable_key)
                }

                #[ic_cdk::update(guard = "is_authenticated")]
                $vis fn [<load_ $name:lower _bytes_from_stable>]() -> Result<(), String> {
                    $name.load_from_stable($stable_key)
                }
            )?
        }
    };
}