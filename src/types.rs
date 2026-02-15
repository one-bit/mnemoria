use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

/// Category tag for a memory entry.
///
/// Entry types help organize memories by their intent. They are stored
/// alongside the entry and can be used for filtering or display purposes.
/// The default type is [`Discovery`](EntryType::Discovery).
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
    /// A goal or intention to accomplish something.
    Intent,
    /// Something learned or observed (default).
    #[default]
    Discovery,
    /// A decision that was made, with rationale.
    Decision,
    /// A problem or issue encountered.
    Problem,
    /// A solution to a previously identified problem.
    Solution,
    /// A recurring pattern worth remembering.
    Pattern,
    /// A warning or caveat for future reference.
    Warning,
    /// A successful outcome or achievement.
    Success,
    /// Notes about a refactoring effort.
    Refactor,
    /// A bug fix that was applied.
    Bugfix,
    /// A feature that was implemented.
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

/// A single memory record stored in the binary log.
///
/// Each entry captures a categorized piece of knowledge with a summary and
/// full content. Entries are linked into a CRC32 checksum chain: each
/// entry's `prev_checksum` references the preceding entry's `checksum`,
/// forming an integrity-verifiable sequence.
///
/// Entries are created via [`Mnemoria::remember`](crate::Mnemoria::remember)
/// and should not normally be constructed directly.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, SerdeSerialize, SerdeDeserialize)]
pub struct MemoryEntry {
    /// Unique identifier (UUID v4 string).
    pub id: String,
    /// Category tag for this memory.
    pub entry_type: EntryType,
    /// Short, human-readable summary of the memory.
    pub summary: String,
    /// Full content of the memory.
    pub content: String,
    /// Optional embedding vector for semantic search (model2vec).
    pub embedding: Option<Vec<f32>>,
    /// Creation timestamp in milliseconds since the Unix epoch.
    pub timestamp: i64,
    /// CRC32 checksum of this entry's fields (including `prev_checksum`).
    pub checksum: u32,
    /// Checksum of the preceding entry in the log (0 for the first entry).
    pub prev_checksum: u32,
}

impl MemoryEntry {
    /// Compute the CRC32 checksum for an entry with the given fields.
    ///
    /// The checksum covers the ID, entry type, summary, content, timestamp,
    /// previous checksum, and embedding (if present). This is used both when
    /// creating new entries and when verifying existing ones.
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

    /// Create a new entry with a generated UUID, current timestamp, and
    /// computed checksum.
    ///
    /// The embedding is initially `None`; it is populated later by
    /// [`Mnemoria::remember`](crate::Mnemoria::remember) if the embedding
    /// backend is available.
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

/// A memory entry paired with its relevance score from a search query.
///
/// Returned by [`Mnemoria::search_memory`](crate::Mnemoria::search_memory).
/// Results are ordered by descending `score`.
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct SearchResult {
    /// The entry's unique ID (same as [`MemoryEntry::id`]).
    pub id: String,
    /// The full memory entry.
    pub entry: MemoryEntry,
    /// Relevance score. Higher is more relevant. When hybrid search is
    /// active, this is the Reciprocal Rank Fusion (RRF) score combining
    /// BM25 and cosine similarity rankings.
    pub score: f32,
}

/// Aggregate statistics about a memory store.
///
/// Returned by [`Mnemoria::memory_stats`](crate::Mnemoria::memory_stats).
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize, Default)]
pub struct MemoryStats {
    /// Total number of entries in the store.
    pub total_entries: u64,
    /// Size of the `log.bin` file in bytes.
    pub file_size_bytes: u64,
    /// Timestamp of the oldest entry (milliseconds since Unix epoch), or
    /// `None` if the store is empty.
    pub oldest_timestamp: Option<i64>,
    /// Timestamp of the newest entry (milliseconds since Unix epoch), or
    /// `None` if the store is empty.
    pub newest_timestamp: Option<i64>,
}

/// Options for retrieving entries in chronological order.
///
/// Used with [`Mnemoria::timeline`](crate::Mnemoria::timeline). The default
/// returns the 20 most recent entries (reverse chronological).
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct TimelineOptions {
    /// Maximum number of entries to return.
    pub limit: usize,
    /// Only include entries at or after this timestamp (milliseconds since
    /// Unix epoch). `None` means no lower bound.
    pub since: Option<i64>,
    /// Only include entries at or before this timestamp (milliseconds since
    /// Unix epoch). `None` means no upper bound.
    pub until: Option<i64>,
    /// If `true`, return entries in reverse chronological order (newest
    /// first). Defaults to `true`.
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

/// Configuration for a memory store.
///
/// Controls durability, entry rotation, and the embedding model. The
/// default configuration uses [`DurabilityMode::Fsync`], no entry limit,
/// and the `"minishlab/potion-base-8M"` embedding model.
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct Config {
    /// Maximum number of entries to keep. When the store exceeds this count,
    /// the oldest entries are automatically rotated out. `None` means no
    /// limit (the default).
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
