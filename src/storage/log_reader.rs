use crate::types::MemoryEntry;
use rkyv::rancor::Error;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub struct RecoveryScan {
    pub entries: Vec<MemoryEntry>,
    pub valid_bytes: u64,
    pub total_bytes: u64,
}

/// Read all decodable entries from the binary log at `path`.
///
/// Parsing stops at the first record that cannot be decoded (e.g. mid-log
/// corruption or a partial trailing write from a crash). Entries *after* the
/// corrupt record are not returned. This is intentional: the append-only log
/// format means corruption only occurs at the tail, and
/// [`scan_recoverable_prefix`] handles truncation during open.
pub fn read_all(path: &Path) -> Result<Vec<MemoryEntry>, crate::Error> {
    let mut entries = Vec::new();

    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
        Err(e) => return Err(e.into()),
    };

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.is_empty() {
        return Ok(entries);
    }

    let mut offset = 0;
    while offset + 4 <= buffer.len() {
        // Safety: the loop guard guarantees at least 4 bytes remain.
        let len = u32::from_le_bytes(
            buffer[offset..offset + 4]
                .try_into()
                .expect("slice is exactly 4 bytes (guarded by loop condition)"),
        ) as usize;
        offset += 4;

        if offset + len > buffer.len() {
            break;
        }

        let data = &buffer[offset..offset + len];

        match rkyv::from_bytes::<MemoryEntry, Error>(data) {
            Ok(entry) => {
                entries.push(entry);
                offset += len;
            }
            Err(e) => {
                tracing::warn!("Failed to decode entry at offset {}: {}", offset, e);
                break;
            }
        }
    }

    Ok(entries)
}

pub fn validate_checksum_chain(path: &Path) -> Result<bool, crate::Error> {
    let entries = read_all(path)?;

    let mut prev_checksum = 0u32;
    for entry in &entries {
        let expected_checksum = MemoryEntry::compute_checksum(
            &entry.id,
            entry.entry_type,
            &entry.summary,
            &entry.content,
            entry.timestamp,
            entry.prev_checksum,
            entry.embedding.as_deref(),
        );

        if entry.prev_checksum != prev_checksum {
            return Ok(false);
        }

        if entry.checksum != expected_checksum {
            return Ok(false);
        }

        prev_checksum = entry.checksum;
    }

    Ok(true)
}

pub fn scan_recoverable_prefix(path: &Path) -> Result<RecoveryScan, crate::Error> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(RecoveryScan {
                entries: Vec::new(),
                valid_bytes: 0,
                total_bytes: 0,
            });
        }
        Err(e) => return Err(e.into()),
    };
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let total_bytes = buffer.len() as u64;
    let mut valid_entries = Vec::new();
    let mut offset = 0usize;
    let mut valid_end = 0usize;
    let mut prev_checksum = 0u32;

    while offset + 4 <= buffer.len() {
        let record_start = offset;
        // Safety: the loop guard guarantees at least 4 bytes remain.
        let len = u32::from_le_bytes(
            buffer[offset..offset + 4]
                .try_into()
                .expect("slice is exactly 4 bytes (guarded by loop condition)"),
        ) as usize;
        offset += 4;

        if offset + len > buffer.len() {
            break;
        }

        let data = &buffer[offset..offset + len];
        let entry = match rkyv::from_bytes::<MemoryEntry, Error>(data) {
            Ok(entry) => entry,
            Err(e) => {
                tracing::warn!(
                    "Failed to decode entry at offset {} during recovery scan: {}",
                    offset,
                    e
                );
                break;
            }
        };

        let expected_checksum = MemoryEntry::compute_checksum(
            &entry.id,
            entry.entry_type,
            &entry.summary,
            &entry.content,
            entry.timestamp,
            entry.prev_checksum,
            entry.embedding.as_deref(),
        );

        if entry.prev_checksum != prev_checksum || entry.checksum != expected_checksum {
            tracing::warn!(
                "Checksum chain mismatch at offset {} during recovery scan",
                record_start
            );
            break;
        }

        prev_checksum = entry.checksum;
        valid_entries.push(entry);
        offset += len;
        valid_end = offset;
    }

    Ok(RecoveryScan {
        entries: valid_entries,
        valid_bytes: valid_end as u64,
        total_bytes,
    })
}

pub fn truncate_to(path: &Path, size: u64) -> Result<(), crate::Error> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(size)?;
    file.sync_all()?;
    Ok(())
}

pub fn file_size(path: &Path) -> Result<u64, crate::Error> {
    let metadata = std::fs::metadata(path)?;
    Ok(metadata.len())
}
