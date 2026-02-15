use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Archive,
    Serialize,
    Deserialize,
    SerdeSerialize,
    SerdeDeserialize,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lower")]
#[derive(Default)]
pub enum EntryType {
    Intent,
    #[default]
    Discovery,
    Decision,
    Problem,
    Solution,
    Pattern,
    Warning,
    Success,
    Refactor,
    Bugfix,
    Feature,
}

impl std::fmt::Display for EntryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntryType::Intent => write!(f, "intent"),
            EntryType::Discovery => write!(f, "discovery"),
            EntryType::Decision => write!(f, "decision"),
            EntryType::Problem => write!(f, "problem"),
            EntryType::Solution => write!(f, "solution"),
            EntryType::Pattern => write!(f, "pattern"),
            EntryType::Warning => write!(f, "warning"),
            EntryType::Success => write!(f, "success"),
            EntryType::Refactor => write!(f, "refactor"),
            EntryType::Bugfix => write!(f, "bugfix"),
            EntryType::Feature => write!(f, "feature"),
        }
    }
}

#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub entry_type: EntryType,
    pub summary: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub timestamp: i64,
    pub checksum: u32,
    pub prev_checksum: u32,
}

impl MemoryEntry {
    pub fn compute_checksum(
        id: &str,
        entry_type: EntryType,
        summary: &str,
        content: &str,
        timestamp: i64,
        prev_checksum: u32,
        embedding: Option<&[f32]>,
    ) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(id.as_bytes());
        hasher.update(&[entry_type as u8]);
        hasher.update(summary.as_bytes());
        hasher.update(content.as_bytes());
        hasher.update(&timestamp.to_le_bytes());
        hasher.update(&prev_checksum.to_le_bytes());

        match embedding {
            Some(values) => {
                hasher.update(&[1]);
                hasher.update(&(values.len() as u32).to_le_bytes());
                for value in values {
                    hasher.update(&value.to_bits().to_le_bytes());
                }
            }
            None => hasher.update(&[0]),
        }

        hasher.finalize()
    }

    pub fn new(
        entry_type: EntryType,
        summary: String,
        content: String,
        prev_checksum: u32,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let checksum = Self::compute_checksum(
            &id,
            entry_type,
            &summary,
            &content,
            timestamp,
            prev_checksum,
            None,
        );

        Self {
            id,
            entry_type,
            summary,
            content,
            embedding: None,
            timestamp,
            checksum,
            prev_checksum,
        }
    }
}

#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct SearchResult {
    pub id: String,
    pub entry: MemoryEntry,
    pub score: f32,
}

#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize, Default)]
pub struct MemoryStats {
    pub total_entries: u64,
    pub file_size_bytes: u64,
    pub oldest_timestamp: Option<i64>,
    pub newest_timestamp: Option<i64>,
}

#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct TimelineOptions {
    pub limit: usize,
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub reverse: bool,
}

impl Default for TimelineOptions {
    fn default() -> Self {
        Self {
            limit: 20,
            since: None,
            until: None,
            reverse: true,
        }
    }
}

/// Controls how aggressively writes are flushed to durable storage.
///
/// Higher durability means lower risk of data loss on crash but reduced
/// write throughput (due to per-entry `fsync`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, SerdeSerialize, SerdeDeserialize, Default)]
pub enum DurabilityMode {
    /// `flush()` + `sync_all()` after every entry write.
    /// Safest: data reaches durable storage before `remember()` returns.
    /// Expect ~100-200 entries/sec on typical SSDs.
    #[default]
    Fsync,

    /// `flush()` only — data reaches the OS page cache but may be lost
    /// on power failure. Good balance of safety and throughput.
    FlushOnly,

    /// No explicit flush or sync. Fastest, but data may be lost on
    /// crash or power failure. Suitable for batch imports or ephemeral data.
    None,
}

#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct Config {
    pub max_entries: Option<u64>,
    /// Controls write durability. See [`DurabilityMode`] for details.
    pub durability: DurabilityMode,
    /// HuggingFace model ID for the model2vec embedding model.
    /// Defaults to `"minishlab/potion-base-8M"`. Any model2vec-compatible
    /// model can be used (e.g. `"minishlab/potion-base-32M"`,
    /// `"minishlab/potion-retrieval-32M"`).
    pub model_id: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_entries: None,
            durability: DurabilityMode::default(),
            model_id: crate::constants::DEFAULT_MODEL_ID.to_string(),
        }
    }
}
