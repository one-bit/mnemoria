use atomic_write_file::AtomicWriteFile;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::constants::{LOG_FILENAME, MANIFEST_FILENAME};

/// On-disk JSON metadata that tracks the state of a memory store.
///
/// The manifest records aggregate counters (entry count, checksum chain
/// head, timestamp bounds) so that the log does not need to be scanned
/// for basic bookkeeping operations. It is atomically rewritten after
/// every mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version (currently always 1).
    pub version: u32,
    /// Total number of entries in the log.
    pub entry_count: u64,
    /// CRC32 checksum of the last entry in the log (0 if empty).
    pub last_checksum: u32,
    /// Timestamp of the oldest entry (ms since epoch), or `None` if empty.
    #[serde(default)]
    pub oldest_timestamp: Option<i64>,
    /// Timestamp of the newest entry (ms since epoch), or `None` if empty.
    #[serde(default)]
    pub newest_timestamp: Option<i64>,
    /// Timestamp when the store was created (ms since epoch).
    pub created_at: i64,
    /// Timestamp of the last modification (ms since epoch).
    pub updated_at: i64,
}

impl Default for Manifest {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        Self {
            version: 1,
            entry_count: 0,
            last_checksum: 0,
            oldest_timestamp: None,
            newest_timestamp: None,
            created_at: now,
            updated_at: now,
        }
    }
}

impl Manifest {
    /// Return the path to `manifest.json` within the given store directory.
    pub fn path(base_path: &Path) -> PathBuf {
        base_path.join(MANIFEST_FILENAME)
    }

    /// Return the path to `log.bin` within the given store directory.
    pub fn log_path(base_path: &Path) -> PathBuf {
        base_path.join(LOG_FILENAME)
    }

    /// Load the manifest from `manifest.json` in the given store directory.
    ///
    /// Returns [`Error::ManifestNotFound`](crate::Error::ManifestNotFound)
    /// if the file does not exist, or
    /// [`Error::ManifestParse`](crate::Error::ManifestParse) if the JSON
    /// is invalid.
    pub fn load(base_path: &Path) -> Result<Self, crate::Error> {
        let path = Self::path(base_path);
        if !path.exists() {
            return Err(crate::Error::ManifestNotFound);
        }

        let content = std::fs::read_to_string(&path)?;

        Ok(serde_json::from_str(&content)?)
    }

    /// Atomically write the manifest to `manifest.json`.
    ///
    /// Uses [`AtomicWriteFile`] to ensure the file is never partially written.
    pub fn save(&self, base_path: &Path) -> Result<(), crate::Error> {
        let path = Self::path(base_path);
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;

        let mut file = AtomicWriteFile::open(&path)?;
        file.write_all(content.as_bytes())?;
        file.commit()?;

        Ok(())
    }
}
