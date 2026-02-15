//! Low-level storage primitives for the append-only binary log.
//!
//! This module contains the binary log reader/writer, manifest management,
//! and advisory file locking. These are internal implementation details;
//! most users should interact through [`Mnemoria`](crate::Mnemoria) instead.

pub mod file_lock;
pub mod log_reader;
pub mod log_writer;
pub mod manifest;

pub use file_lock::FileLock;
pub use log_writer::LogWriter;
pub use manifest::Manifest;
