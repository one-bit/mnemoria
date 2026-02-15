use std::sync::{Mutex, MutexGuard};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Manifest not found")]
    ManifestNotFound,

    #[error("Failed to parse manifest: {0}")]
    ManifestParse(#[from] serde_json::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Search error: {0}")]
    Search(#[from] tantivy::TantivyError),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Lock error")]
    Lock,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Lock a mutex, converting a poisoned-lock panic into `Error::Lock`.
pub fn lock_mutex<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex.lock().map_err(|_| Error::Lock)
}
