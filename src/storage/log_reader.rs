use crate::types::MemoryEntry;
use rkyv::rancor::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug)]
pub struct RecoveryScan {
    pub entries: Vec<MemoryEntry>,
    pub valid_bytes: u64,
    pub total_bytes: u64,
}

pub fn read_all(path: &Path) -> Result<Vec<MemoryEntry>, crate::Error> {
    let mut entries = Vec::new();

    if !path.exists() {
        return Ok(entries);
    }

    let mut file = File::open(path)?;

    file.seek(SeekFrom::Start(0))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.is_empty() {
        return Ok(entries);
    }

    let mut offset = 0;
    while offset + 4 <= buffer.len() {
        let len = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
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
    if !path.exists() {
        return Ok(RecoveryScan {
            entries: Vec::new(),
            valid_bytes: 0,
            total_bytes: 0,
        });
    }

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let total_bytes = buffer.len() as u64;
    let mut valid_entries = Vec::new();
    let mut offset = 0usize;
    let mut valid_end = 0usize;
    let mut prev_checksum = 0u32;

    while offset + 4 <= buffer.len() {
        let record_start = offset;
        let len = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
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
