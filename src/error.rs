use std::sync::{Mutex, MutexGuard};
use thiserror::Error;

/// Errors that can occur when interacting with a memory store.
#[derive(Error, Debug)]
pub enum Error {
    /// An I/O error occurred (file read/write, directory creation, etc.).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The `manifest.json` file was not found in the memory store directory.
    ///
    /// This typically means the path does not point to an initialized store.
    /// During [`open`](crate::Mnemoria::open), the manifest is automatically
    /// rebuilt from the log if missing.
    #[error("Manifest not found")]
    ManifestNotFound,

    /// The `manifest.json` file exists but contains invalid JSON.
    ///
    /// During [`open`](crate::Mnemoria::open), a corrupt manifest is
    /// automatically rebuilt from the log.
    #[error("Failed to parse manifest: {0}")]
    ManifestParse(#[from] serde_json::Error),

    /// A serialization or deserialization error (rkyv or JSON).
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// An error from the Tantivy full-text search engine.
    #[error("Search error: {0}")]
    Search(#[from] tantivy::TantivyError),

    /// An error from the embedding backend (model loading or inference).
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// An internal mutex was poisoned (a thread panicked while holding it).
    #[error("Lock error")]
    Lock,
}

/// A specialized [`Result`](std::result::Result) type for mnemoria operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Lock a mutex, converting a poisoned-lock panic into `Error::Lock`.
pub fn lock_mutex<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    mutex.lock().map_err(|_| Error::Lock)
}
