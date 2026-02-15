use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::constants::{LOG_FILENAME, MANIFEST_FILENAME};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub entry_count: u64,
    pub last_checksum: u32,
    #[serde(default)]
    pub oldest_timestamp: Option<i64>,
    #[serde(default)]
    pub newest_timestamp: Option<i64>,
    pub created_at: i64,
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
    pub fn path(base_path: &Path) -> PathBuf {
        base_path.join(MANIFEST_FILENAME)
    }

    pub fn log_path(base_path: &Path) -> PathBuf {
        base_path.join(LOG_FILENAME)
    }

    pub fn load(base_path: &Path) -> Result<Self, crate::Error> {
        let path = Self::path(base_path);
        if !path.exists() {
            return Err(crate::Error::ManifestNotFound);
        }

        let content = std::fs::read_to_string(&path)?;

        Ok(serde_json::from_str(&content)?)
    }

    pub fn save(&self, base_path: &Path) -> Result<(), crate::Error> {
        let path = Self::path(base_path);
        let temp_path = base_path.join(format!("{MANIFEST_FILENAME}.tmp"));
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;

        let write_result = (|| -> Result<(), crate::Error> {
            let mut temp_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&temp_path)?;

            temp_file.write_all(content.as_bytes())?;
            temp_file.flush()?;
            temp_file.sync_all()?;
            Ok(())
        })();

        if let Err(e) = write_result {
            // Best-effort cleanup of the temp file on write failure.
            let _ = std::fs::remove_file(&temp_path);
            return Err(e);
        }

        std::fs::rename(&temp_path, &path)?;

        let dir_file = OpenOptions::new().read(true).open(base_path)?;
        dir_file.sync_all()?;
        Ok(())
    }
}
