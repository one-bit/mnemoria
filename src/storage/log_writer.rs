use crate::types::{DurabilityMode, MemoryEntry};
use rkyv::rancor::Error;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

pub struct LogWriter {
    file: File,
    durability: DurabilityMode,
}

impl LogWriter {
    pub fn new(path: &Path) -> Result<Self, crate::Error> {
        Self::with_durability(path, DurabilityMode::Fsync)
    }

    pub fn with_durability(path: &Path, durability: DurabilityMode) -> Result<Self, crate::Error> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;

        Ok(Self { file, durability })
    }

    pub fn append(&mut self, entry: &MemoryEntry) -> Result<(), crate::Error> {
        let encoded = rkyv::to_bytes::<Error>(entry)
            .map_err(|e: Error| crate::Error::Serialization(e.to_string()))?;

        // Build a single contiguous buffer (length prefix + payload) so the
        // write is a single syscall. This prevents interleaving even if
        // advisory locking is somehow bypassed.
        let len = encoded.len() as u32;
        let mut buf = Vec::with_capacity(4 + encoded.len());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&encoded);

        self.file.write_all(&buf)?;

        match self.durability {
            DurabilityMode::Fsync => {
                self.file.flush()?;
                self.file.sync_all()?;
            }
            DurabilityMode::FlushOnly => {
                self.file.flush()?;
            }
            DurabilityMode::None => {}
        }

        self.file.seek(SeekFrom::End(0))?;

        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), crate::Error> {
        self.file.flush()?;
        Ok(())
    }
}
