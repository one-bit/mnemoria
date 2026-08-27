use crate::embeddings::EmbeddingBackend;
use crate::error::lock_mutex;
use crate::search::IndexManager;
use crate::storage::file_lock::FileLock;
use crate::storage::log_reader;
use crate::storage::{LogWriter, Manifest};
use crate::types::{
    AskOptions, CompactOptions, CompactReport, Config, ConsolidateReport, EntryType, MemoryEntry,
    MemoryStats, SearchResult, TimelineOptions,
};
use atomic_write_file::AtomicWriteFile;
use rkyv::rancor::Error as RkyvError;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const INDEX_COMMIT_BATCH_SIZE: usize = 32;
const INDEX_DIR_PREFIX: &str = "mnemoria-idx-";

/// Truncate a string to at most `max_bytes` bytes, ensuring the cut happens
/// at a valid UTF-8 char boundary. Returns the full string if it is already
/// within the limit.
fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Prefix used by tombstone entries created via `forget`-style supersession.
///
/// A tombstone entry's summary has the form `SUPERSEDES <uuid> — <reason>`
/// (em dash U+2014). Tombstones are append-only audit markers: they record
/// that the referenced entry is obsolete without deleting it.
pub(crate) const SUPERSEDES_PREFIX: &str = "SUPERSEDES ";

/// Extract the superseded entry id from a tombstone summary, if present.
///
/// Returns the first UUID-shaped token (8-4-4-4-12 hex groups) following the
/// `SUPERSEDES ` prefix, lowercased for stable comparison. Returns `None` if
/// the summary is not a tombstone or contains no parseable UUID.
fn parse_superseded_id(summary: &str) -> Option<String> {
    let rest = summary.strip_prefix(SUPERSEDES_PREFIX)?;
    let token = rest.split_whitespace().next()?;
    let bytes = token.as_bytes();
    if bytes.len() != 36 {
        return None;
    }
    let group_lens = [8usize, 4, 4, 4, 12];
    let mut pos = 0usize;
    for (i, &len) in group_lens.iter().enumerate() {
        for c in &bytes[pos..pos + len] {
            if !c.is_ascii_hexdigit() {
                return None;
            }
        }
        pos += len;
        if i < group_lens.len() - 1 {
            if bytes[pos] != b'-' {
                return None;
            }
            pos += 1;
        }
    }
    Some(token.to_ascii_lowercase())
}

/// Collect the set of entry ids that have been superseded by tombstones.
///
/// Scans the cache for entries whose summary starts with [`SUPERSEDES_PREFIX`]
/// and returns the parsed target ids. Superseded entries are hidden from
/// search results (their tombstones are hidden too), so recall surfaces only
/// currently-valid memories.
fn superseded_ids(cache: &crate::api::EntryCache) -> HashSet<String> {
    let mut superseded = HashSet::new();
    for entry in &cache.ordered {
        if let Some(target) = parse_superseded_id(&entry.summary) {
            superseded.insert(target);
        }
    }
    superseded
}

/// Half-life (in days) of the exponential recency decay applied to search
/// scores. An entry this old keeps roughly the midpoint between 1.0 and
/// [`RECENCY_FLOOR`].
const RECENCY_HALF_LIFE_DAYS: f64 = 120.0;

/// Minimum recency factor: no matter how old an entry is, its score keeps
/// at least this fraction of the recency component. Durable knowledge must
/// remain retrievable, only gradually deprioritized.
const RECENCY_FLOOR: f64 = 0.6;

/// Relative score floor: results whose adjusted score falls below this
/// fraction of the best result's adjusted score are dropped, so weak tail
/// matches do not surface as noise.
const MIN_SCORE_RATIO: f32 = 0.2;

/// Filename of the append-only usage-event sidecar (F6 usage tracking).
///
/// Kept separate from `log.bin` so the entry log stays byte-compatible
/// with upstream 0.3.5 stores; usage events are advisory signals, not
/// memory entries.
const USAGE_LOG_FILE: &str = "usage.jsonl";

/// Per-use score boost. Each recorded use adds this multiplier, up to
/// `USAGE_COUNT_CAP` uses.
const USAGE_BOOST_PER_USE: f32 = 0.1;

/// Number of uses after which the usage boost saturates. With the defaults
/// the maximum usage multiplier is `1.0 + 5 * 0.1 = 1.5`.
const USAGE_COUNT_CAP: u32 = 5;

/// Usage multiplier in `[1.0, 1.0 + USAGE_COUNT_CAP * USAGE_BOOST_PER_USE]`.
///
/// Entries that agents repeatedly mark as useful earn a bounded ranking
/// boost; never-used entries keep a neutral factor of 1.0.
fn usage_factor(use_count: u32) -> f32 {
    1.0 + USAGE_BOOST_PER_USE * use_count.min(USAGE_COUNT_CAP) as f32
}

/// One append-only usage event (F6 usage tracking).
///
/// Serialized as one JSON object per line in `usage.jsonl`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct UsageEvent {
    /// Id of the entry that was marked useful.
    id: String,
    /// Unix milliseconds when the use was recorded.
    timestamp: i64,
}

/// Path of the usage-event sidecar for a store.
fn usage_log_path(base_path: &Path) -> PathBuf {
    base_path.join(USAGE_LOG_FILE)
}

/// Load per-entry usage counts from `usage.jsonl`.
///
/// The sidecar is advisory: a missing file yields an empty map, and a
/// corrupt or partially written trailing line is skipped rather than
/// treated as an error.
fn load_usage_counts(base_path: &Path) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let path = usage_log_path(base_path);
    let Ok(content) = std::fs::read_to_string(&path) else {
        return counts;
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<UsageEvent>(line) {
            *counts.entry(event.id).or_insert(0) += 1;
        }
    }
    counts
}

/// Minimum length for an identifier-style entity token to be indexed.
const MIN_ENTITY_LEN: usize = 3;

/// Extract mechanical entity tokens from an entry's summary and content (F7).
///
/// No LLM or NLP: this is a deterministic scanner that recognizes three
/// token classes worth remembering across entries:
///
/// 1. UUIDs (8-4-4-4-12 hex groups), lowercased.
/// 2. File paths (anything containing a path separator or a drive letter).
/// 3. Identifier-style tokens containing a hyphen, underscore, or dot
///    (e.g. crate names, config keys, version strings), at least
///    MIN_ENTITY_LEN chars.
///
/// All tokens are lowercased so lookups are case-insensitive.
fn extract_entities(summary: &str, content: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for text in [summary, content] {
        for token in scan_entity_tokens(text) {
            out.insert(token);
        }
    }
    out
}

/// Scan one text blob for entity tokens.
fn scan_entity_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        // UUID: 8-4-4-4-12 hex groups.
        if c.is_ascii_hexdigit()
            && let Some((uuid, consumed)) = try_scan_uuid(&chars, i)
        {
            tokens.push(uuid.to_ascii_lowercase());
            i += consumed;
            continue;
        }

        // Path: starts with a drive letter (X:) or a separator.
        if (c.is_ascii_alphabetic()
            && i + 1 < len
            && chars[i + 1] == ':'
            && i + 2 < len
            && (chars[i + 2] == '/' || chars[i + 2] == '\\'))
            || c == '/'
            || c == '\\'
        {
            let (path, consumed) = scan_path(&chars, i);
            if consumed > 0 {
                tokens.push(path.to_ascii_lowercase());
                i += consumed;
                continue;
            }
        }

        // Identifier-style token: alphanumeric run that contains at least
        // one hyphen, underscore, or dot.
        if c.is_ascii_alphanumeric() {
            let (token, consumed) = scan_identifier(&chars, i);
            if token.len() >= MIN_ENTITY_LEN
                && (token.contains('-') || token.contains('_') || token.contains('.'))
            {
                tokens.push(token.to_ascii_lowercase());
            }
            i += consumed;
            continue;
        }

        i += 1;
    }

    tokens
}

/// Try to scan a UUID (8-4-4-4-12 hex) starting at position i.
/// Returns the matched string and number of chars consumed, or None.
fn try_scan_uuid(chars: &[char], i: usize) -> Option<(String, usize)> {
    let groups = [8, 4, 4, 4, 12];
    let mut pos = i;
    let mut matched = String::new();

    for (g, &width) in groups.iter().enumerate() {
        for _ in 0..width {
            if pos < chars.len() && chars[pos].is_ascii_hexdigit() {
                matched.push(chars[pos]);
                pos += 1;
            } else {
                return None;
            }
        }
        if g < groups.len() - 1 {
            if pos < chars.len() && chars[pos] == '-' {
                matched.push('-');
                pos += 1;
            } else {
                return None;
            }
        }
    }

    Some((matched, pos - i))
}

/// Scan a file path starting at position i. Consumes until whitespace or
/// a closing bracket/quote. Returns the path and chars consumed.
fn scan_path(chars: &[char], i: usize) -> (String, usize) {
    let mut pos = i;
    let mut path = String::new();

    while pos < chars.len() {
        let c = chars[pos];
        if c.is_whitespace()
            || c == '"'
            || c == '\''
            || c == ')'
            || c == ']'
            || c == '}'
            || c == ','
            || c == ';'
        {
            break;
        }
        path.push(c);
        pos += 1;
    }

    // Strip trailing punctuation that is unlikely part of the path.
    while path.ends_with('.') || path.ends_with(':') || path.ends_with(')') {
        path.pop();
        pos -= 1;
    }

    (path, pos - i)
}

/// Scan an identifier-style token (alphanumeric plus hyphen/underscore/dot)
/// starting at position i. Returns the token and chars consumed.
fn scan_identifier(chars: &[char], i: usize) -> (String, usize) {
    let mut pos = i;
    let mut token = String::new();

    while pos < chars.len() {
        let c = chars[pos];
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
            token.push(c);
            pos += 1;
        } else {
            break;
        }
    }

    // Strip trailing dots and hyphens (sentence punctuation). The caller
    // only invokes this on an alphanumeric start char, so no leading
    // punctuation can appear. All scanned chars are consumed.
    let trimmed = token.trim_end_matches(['.', '-']).to_string();
    (trimmed, pos - i)
}

/// Importance multiplier per entry type.
///
/// Durable, high-signal categories (warnings, decisions, solutions) rank
/// above the discovery baseline; transient intentions rank slightly below.
fn type_weight(entry_type: EntryType) -> f32 {
    match entry_type {
        EntryType::Warning => 1.2,
        EntryType::Decision | EntryType::Solution => 1.15,
        EntryType::Bugfix | EntryType::Pattern => 1.1,
        EntryType::Feature => 1.05,
        EntryType::Intent => 0.9,
        EntryType::Discovery | EntryType::Problem | EntryType::Success | EntryType::Refactor => 1.0,
    }
}

/// Exponential recency factor in `[RECENCY_FLOOR, 1.0]` based on entry age.
///
/// Future timestamps (clock skew) clamp to a factor of 1.0.
fn recency_factor(timestamp_ms: i64, now_ms: i64) -> f32 {
    let age_days = (now_ms - timestamp_ms).max(0) as f64 / 86_400_000.0;
    let decay = (-age_days / RECENCY_HALF_LIFE_DAYS).exp();
    (RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * decay) as f32
}

/// Blend a raw retrieval score with entry-type importance, recency, and
/// recorded usage (F6).
fn adjust_score(raw: f32, entry: &MemoryEntry, now_ms: i64, use_count: u32) -> f32 {
    raw * type_weight(entry.entry_type)
        * recency_factor(entry.timestamp, now_ms)
        * usage_factor(use_count)
}

/// CRC32 hash of the canonical base path, used to scope ephemeral index
/// directories to a specific memory store.
fn base_path_hash(base_path: &Path) -> u32 {
    let canonical = base_path
        .canonicalize()
        .unwrap_or_else(|_| base_path.to_path_buf());
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(canonical.to_string_lossy().as_bytes());
    hasher.finalize()
}

/// Directory name prefix shared by all ephemeral index dirs for a given memory store.
fn index_dir_prefix(base_path: &Path) -> String {
    format!("{INDEX_DIR_PREFIX}{:08x}-", base_path_hash(base_path))
}

/// Generate a per-process index directory under the OS temp directory.
/// Each process gets its own Tantivy index (since Tantivy requires exclusive
/// write access). The index is ephemeral -- rebuilt from `log.bin` on open.
///
/// Path includes a CRC32 of the base path for uniqueness across different
/// memory stores, plus PID and a random suffix for uniqueness across
/// processes sharing the same store.
fn per_process_index_path(base_path: &Path) -> PathBuf {
    let pid = std::process::id();
    let random = uuid::Uuid::new_v4().as_u128() & 0xFFFF_FFFF;
    let prefix = index_dir_prefix(base_path);

    std::env::temp_dir().join(format!("{prefix}{pid}-{random:08x}"))
}

/// Remove ephemeral index directories left behind by crashed processes.
///
/// Scans the OS temp directory for dirs matching this memory store's prefix,
/// extracts the PID from each directory name, and removes any whose PID is
/// no longer alive. Errors are silently ignored (best-effort cleanup).
fn cleanup_stale_index_dirs(base_path: &Path) {
    let prefix = index_dir_prefix(base_path);
    let tmp_dir = std::env::temp_dir();

    let entries = match std::fs::read_dir(&tmp_dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if !name_str.starts_with(&prefix) {
            continue;
        }

        // Format: "mnemoria-idx-{hash}-{pid}-{random}"
        // After stripping the prefix we have "{pid}-{random}"
        let remainder = &name_str[prefix.len()..];
        let pid_str = match remainder.split('-').next() {
            Some(s) => s,
            None => continue,
        };
        let pid: u32 = match pid_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        if !is_process_alive(pid) {
            let dir_path = tmp_dir.join(&*name_str);
            let _ = std::fs::remove_dir_all(&dir_path);
        }
    }
}

/// Check whether a process with the given PID is still running.
#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    // Reject PIDs that would overflow i32 (would be interpreted as a
    // process group on negative values).
    let pid_i32 = match i32::try_from(pid) {
        Ok(p) if p > 0 => p,
        _ => return false,
    };

    // kill(pid, 0) checks existence without sending a signal.
    // Returns 0 if the process exists and we have permission to signal it.
    // Returns -1 with ESRCH if no such process exists.
    // Returns -1 with EPERM if the process exists but we lack permission.
    // SAFETY: kill(2) with signal 0 is safe for any pid.
    let ret = unsafe { libc::kill(pid_i32, 0) };
    if ret == 0 {
        return true;
    }
    // EPERM means the process exists but we can't signal it — still alive.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
fn is_process_alive(pid: u32) -> bool {
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;

    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut core::ffi::c_void;
        fn CloseHandle(handle: *mut core::ffi::c_void) -> i32;
        fn GetExitCodeProcess(handle: *mut core::ffi::c_void, code: *mut u32) -> i32;
    }

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            // Can't open the process -- it either doesn't exist or we lack
            // permission. In both cases, treat as dead (safe to clean up).
            return false;
        }

        let mut exit_code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut exit_code);
        CloseHandle(handle);

        ok != 0 && exit_code == STILL_ACTIVE
    }
}

struct EntryCache {
    ordered: Vec<MemoryEntry>,
    by_id: HashMap<String, MemoryEntry>,
    /// Inverted entity index (F7): normalized entity token -> entry ids.
    entities: HashMap<String, HashSet<String>>,
}

struct OpenReconciliation {
    manifest: Manifest,
    entries: Vec<MemoryEntry>,
}

impl EntryCache {
    fn from_entries(entries: Vec<MemoryEntry>) -> Self {
        let mut by_id = HashMap::new();
        let mut entities: HashMap<String, HashSet<String>> = HashMap::new();

        for entry in &entries {
            by_id.insert(entry.id.clone(), entry.clone());
            for token in extract_entities(&entry.summary, &entry.content) {
                entities.entry(token).or_default().insert(entry.id.clone());
            }
        }

        Self {
            ordered: entries,
            by_id,
            entities,
        }
    }

    fn empty() -> Self {
        Self::from_entries(Vec::new())
    }

    fn replace(&mut self, entries: Vec<MemoryEntry>) {
        *self = Self::from_entries(entries);
    }

    fn push(&mut self, entry: MemoryEntry) {
        for token in extract_entities(&entry.summary, &entry.content) {
            self.entities
                .entry(token)
                .or_default()
                .insert(entry.id.clone());
        }
        self.by_id.insert(entry.id.clone(), entry.clone());
        self.ordered.push(entry);
    }
}

/// Snapshot of manifest state used to detect when another process has written
/// to the memory store, so we know to reload our in-memory cache.
#[derive(Clone, PartialEq)]
struct ManifestFingerprint {
    entry_count: u64,
    last_checksum: u32,
    updated_at: i64,
}

impl ManifestFingerprint {
    fn from_manifest(m: &Manifest) -> Self {
        Self {
            entry_count: m.entry_count,
            last_checksum: m.last_checksum,
            updated_at: m.updated_at,
        }
    }
}

/// The main API for interacting with a mnemoria memory store.
///
/// `Mnemoria` manages an append-only binary log of [`MemoryEntry`] records,
/// a BM25 full-text search index (via Tantivy), and optional semantic
/// embeddings (via model2vec). It supports concurrent access from multiple
/// processes through advisory file locking and per-process ephemeral indexes.
///
/// # Creating vs. opening
///
/// Use [`Mnemoria::create`] (or [`Mnemoria::create_with_config`]) to
/// initialize a new, empty memory store at a given path. Use
/// [`Mnemoria::open`] (or [`Mnemoria::open_with_config`]) to open an
/// existing store. Opening performs automatic crash recovery: partial
/// trailing writes are truncated and the manifest is reconciled with the
/// log contents.
///
/// # Thread safety
///
/// All public methods take `&self` and use internal [`Mutex`] locks, so a
/// single `Mnemoria` instance can be shared across threads (e.g. via
/// `Arc<Mnemoria>`). Cross-process coordination uses advisory file locks.
///
/// # Cleanup
///
/// The per-process Tantivy index directory (stored in the OS temp dir) is
/// automatically removed when the `Mnemoria` instance is dropped.
/// ## Lock ordering
///
/// When acquiring multiple mutexes, always follow this order to prevent
/// deadlocks:
///
/// 1. `writer`
/// 2. `manifest`
/// 3. `index`
/// 4. `pending_index_writes`
/// 5. `cache`
/// 6. `cached_fingerprint`
///
/// Not every method needs all locks. The rule is: if you hold lock N, you
/// must not attempt to acquire lock M where M < N.
pub struct Mnemoria {
    base_path: PathBuf,
    config: Config,
    // See lock ordering above. Fields are listed in acquisition order.
    writer: Mutex<Option<LogWriter>>,
    manifest: Mutex<Manifest>,
    index: Mutex<IndexManager>,
    /// Per-process index directory (ephemeral, cleaned up on drop).
    index_path: PathBuf,
    pending_index_writes: Mutex<usize>,
    cache: Mutex<EntryCache>,
    embeddings: EmbeddingBackend,
    file_lock: FileLock,
    /// Fingerprint of the manifest as of our last load/write. If the on-disk
    /// manifest differs, another process has written and we must reload.
    cached_fingerprint: Mutex<ManifestFingerprint>,
    /// Per-entry usage counts loaded from `usage.jsonl` (F6 usage tracking).
    usage_counts: Mutex<std::collections::HashMap<String, u32>>,
}

impl Drop for Mnemoria {
    fn drop(&mut self) {
        // Release the Tantivy writer before removing the directory.
        // We take the writer out so its file handles are closed.
        if let Ok(mut index) = self.index.lock() {
            index.drop_writer();
        }
        if self.index_path.exists() {
            let _ = std::fs::remove_dir_all(&self.index_path);
        }
    }
}

impl Mnemoria {
    fn now_millis() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }

    fn timestamp_bounds(entries: &[MemoryEntry]) -> (Option<i64>, Option<i64>) {
        let mut oldest: Option<i64> = None;
        let mut newest: Option<i64> = None;

        for entry in entries {
            oldest = Some(match oldest {
                Some(current) => current.min(entry.timestamp),
                None => entry.timestamp,
            });
            newest = Some(match newest {
                Some(current) => current.max(entry.timestamp),
                None => entry.timestamp,
            });
        }

        (oldest, newest)
    }

    fn manifest_matches_entries(manifest: &Manifest, entries: &[MemoryEntry]) -> bool {
        let (oldest, newest) = Self::timestamp_bounds(entries);
        let last_checksum = entries.last().map_or(0, |entry| entry.checksum);

        manifest.entry_count == entries.len() as u64
            && manifest.last_checksum == last_checksum
            && manifest.oldest_timestamp == oldest
            && manifest.newest_timestamp == newest
    }

    fn manifest_from_entries(existing: Option<&Manifest>, entries: &[MemoryEntry]) -> Manifest {
        let (oldest, newest) = Self::timestamp_bounds(entries);
        let last_checksum = entries.last().map_or(0, |entry| entry.checksum);
        let now = Self::now_millis();

        let mut manifest = existing.cloned().unwrap_or_default();
        if existing.is_none() {
            manifest.created_at = now;
        }

        manifest.entry_count = entries.len() as u64;
        manifest.last_checksum = last_checksum;
        manifest.oldest_timestamp = oldest;
        manifest.newest_timestamp = newest;

        manifest
    }

    fn reconcile_state_on_open(path: &Path) -> Result<OpenReconciliation, crate::Error> {
        let log_path = Manifest::log_path(path);
        let scan = log_reader::scan_recoverable_prefix(&log_path)?;

        if scan.valid_bytes < scan.total_bytes {
            tracing::warn!(
                "Truncating log tail from {} to {} bytes during open() reconciliation",
                scan.total_bytes,
                scan.valid_bytes
            );
            log_reader::truncate_to(&log_path, scan.valid_bytes)?;
        }

        let existing_manifest = match Manifest::load(path) {
            Ok(manifest) => Some(manifest),
            Err(crate::Error::ManifestNotFound | crate::Error::ManifestParse(_)) => None,
            Err(err) => return Err(err),
        };

        let mut reconciled_manifest =
            Self::manifest_from_entries(existing_manifest.as_ref(), &scan.entries);

        let needs_manifest_repair = existing_manifest
            .as_ref()
            .is_none_or(|manifest| !Self::manifest_matches_entries(manifest, &scan.entries));

        if needs_manifest_repair {
            reconciled_manifest.updated_at = Self::now_millis();
            reconciled_manifest.save(path)?;
        }

        Ok(OpenReconciliation {
            manifest: reconciled_manifest,
            entries: scan.entries,
        })
    }

    fn rewrite_log_atomically(
        &self,
        rewritten_entries: &[MemoryEntry],
    ) -> Result<(), crate::Error> {
        let log_path = Manifest::log_path(&self.base_path);

        {
            let mut writer = lock_mutex(&self.writer)?;
            *writer = None;
        }

        {
            let mut file = AtomicWriteFile::open(&log_path)?;
            for entry in rewritten_entries {
                let encoded = rkyv::to_bytes::<RkyvError>(entry)
                    .map_err(|e: RkyvError| crate::Error::Serialization(e.to_string()))?;
                let len = encoded.len() as u32;
                file.write_all(&len.to_le_bytes())?;
                file.write_all(&encoded)?;
            }
            file.commit()?;
        }

        let mut writer = lock_mutex(&self.writer)?;
        *writer = Some(LogWriter::with_durability(
            &log_path,
            self.config.durability,
        )?);

        Ok(())
    }

    fn commit_pending_index_writes(&self, force: bool) -> Result<(), crate::Error> {
        let mut index = lock_mutex(&self.index)?;
        let mut pending = lock_mutex(&self.pending_index_writes)?;

        if *pending == 0 {
            return Ok(());
        }

        if force || *pending >= INDEX_COMMIT_BATCH_SIZE {
            index.commit()?;
            *pending = 0;
        }

        Ok(())
    }

    fn reset_pending_index_writes(&self) -> Result<(), crate::Error> {
        let mut pending = lock_mutex(&self.pending_index_writes)?;
        *pending = 0;
        Ok(())
    }

    /// Update our cached fingerprint to match the current in-memory manifest.
    /// Call this after every successful write operation.
    fn update_fingerprint(&self) -> Result<(), crate::Error> {
        let manifest = lock_mutex(&self.manifest)?;
        let mut fp = lock_mutex(&self.cached_fingerprint)?;
        *fp = ManifestFingerprint::from_manifest(&manifest);
        Ok(())
    }

    /// Check if the on-disk manifest has been modified by another process.
    /// If so, reload the log, manifest, cache, and index from disk.
    /// This must be called while holding at least a shared file lock.
    fn refresh_if_stale(&self) -> Result<(), crate::Error> {
        let disk_manifest = match Manifest::load(&self.base_path) {
            Ok(m) => m,
            Err(crate::Error::ManifestNotFound | crate::Error::ManifestParse(_)) => {
                return Ok(());
            }
            Err(e) => return Err(e),
        };

        let disk_fp = ManifestFingerprint::from_manifest(&disk_manifest);
        let is_stale = {
            let cached_fp = lock_mutex(&self.cached_fingerprint)?;
            *cached_fp != disk_fp
        };

        if !is_stale {
            return Ok(());
        }

        tracing::info!("Detected external modification, reloading from disk");

        // Reload entries from the log
        let log_path = Manifest::log_path(&self.base_path);
        let entries = log_reader::read_all(&log_path)?;

        // Each lock is acquired and released in its own scope (no two held
        // simultaneously), but we follow the documented acquisition order
        // for readability: writer -> manifest -> index -> pending -> cache
        // -> fingerprint.

        // Reopen the log writer (file position may have changed)
        {
            let mut writer = lock_mutex(&self.writer)?;
            *writer = Some(LogWriter::with_durability(
                &log_path,
                self.config.durability,
            )?);
        }

        // Update manifest
        {
            let mut manifest = lock_mutex(&self.manifest)?;
            *manifest = disk_manifest;
        }

        // Rebuild the search index
        {
            let mut index = lock_mutex(&self.index)?;
            index.rebuild_from_entries(&entries)?;
        }
        self.reset_pending_index_writes()?;

        // Update cache and fingerprint
        {
            let mut cache = lock_mutex(&self.cache)?;
            cache.replace(entries);
        }
        {
            let mut fp = lock_mutex(&self.cached_fingerprint)?;
            *fp = disk_fp;
        }

        Ok(())
    }

    /// Create a new memory store at `path` with default configuration.
    ///
    /// The directory is created if it does not exist. An empty `log.bin` and
    /// `manifest.json` are written. If a store already exists at `path`, it
    /// is fully reset: the log is truncated and the manifest is overwritten.
    ///
    /// This is equivalent to calling
    /// [`create_with_config`](Self::create_with_config) with
    /// [`Config::default()`].
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or the initial
    /// files cannot be written.
    pub async fn create(path: &Path) -> Result<Self, crate::Error> {
        Self::create_with_config(path, Config::default()).await
    }

    /// Create a new memory store at `path` with the given [`Config`].
    ///
    /// See [`create`](Self::create) for details. The `config` parameter
    /// allows you to set the durability mode, maximum entry count, and
    /// embedding model.
    pub async fn create_with_config(path: &Path, config: Config) -> Result<Self, crate::Error> {
        std::fs::create_dir_all(path)?;

        let file_lock = FileLock::new(path)?;
        let _guard = file_lock.lock_exclusive()?;

        cleanup_stale_index_dirs(path);

        let manifest = Manifest::default();
        manifest.save(path)?;

        // Truncate any existing log file so the empty manifest and empty log
        // are consistent. Without this, a pre-existing log would contain
        // entries that the fresh manifest doesn't know about, breaking the
        // checksum chain on the next write.
        let log_path = Manifest::log_path(path);
        if log_path.exists() {
            std::fs::write(&log_path, b"")?;
        }

        // A fresh store has no usage history either.
        let usage_path = usage_log_path(path);
        if usage_path.exists() {
            std::fs::write(&usage_path, b"")?;
        }

        let writer = LogWriter::with_durability(&log_path, config.durability)?;

        let fingerprint = ManifestFingerprint::from_manifest(&manifest);

        let index_path = per_process_index_path(path);
        let index = IndexManager::new(&index_path)?;

        let embeddings = EmbeddingBackend::new(&config.model_id);

        Ok(Self {
            base_path: path.to_path_buf(),
            config,
            manifest: Mutex::new(manifest),
            writer: Mutex::new(Some(writer)),
            index: Mutex::new(index),
            index_path,
            pending_index_writes: Mutex::new(0),
            cache: Mutex::new(EntryCache::empty()),
            embeddings,
            file_lock,
            cached_fingerprint: Mutex::new(fingerprint),
            usage_counts: Mutex::new(load_usage_counts(path)),
        })
    }

    /// Open an existing memory store at `path` with the given [`Config`].
    ///
    /// On open, the log is scanned and reconciled with the manifest:
    ///
    /// - Partial trailing records (from a crash) are truncated.
    /// - If the manifest is missing or corrupt, it is rebuilt from the log.
    /// - The full-text search index is rebuilt from the log entries.
    /// - Stale ephemeral index directories from crashed processes are cleaned
    ///   up.
    ///
    /// # Errors
    ///
    /// Returns an error if `path` does not exist or the log cannot be read.
    pub async fn open_with_config(path: &Path, config: Config) -> Result<Self, crate::Error> {
        let file_lock = FileLock::new(path)?;
        // Exclusive lock during open: reconciliation may truncate the log and rewrite the manifest.
        let _guard = file_lock.lock_exclusive()?;

        cleanup_stale_index_dirs(path);

        let open_state = Self::reconcile_state_on_open(path)?;
        let manifest = open_state.manifest;
        let existing_entries = open_state.entries;
        let log_path = Manifest::log_path(path);

        // Each process gets its own Tantivy index directory (ephemeral).
        let index_path = per_process_index_path(path);
        let mut index = IndexManager::new(&index_path)?;
        index.rebuild_from_entries(&existing_entries)?;

        let fingerprint = ManifestFingerprint::from_manifest(&manifest);

        let embeddings = EmbeddingBackend::new(&config.model_id);

        Ok(Self {
            base_path: path.to_path_buf(),
            config: config.clone(),
            manifest: Mutex::new(manifest),
            writer: Mutex::new(Some(LogWriter::with_durability(
                &log_path,
                config.durability,
            )?)),
            index: Mutex::new(index),
            index_path,
            pending_index_writes: Mutex::new(0),
            cache: Mutex::new(EntryCache::from_entries(existing_entries)),
            embeddings,
            file_lock,
            cached_fingerprint: Mutex::new(fingerprint),
            usage_counts: Mutex::new(load_usage_counts(path)),
        })
    }

    /// Open an existing memory store at `path` with default configuration.
    ///
    /// This is equivalent to calling
    /// [`open_with_config`](Self::open_with_config) with
    /// [`Config::default()`].
    ///
    /// See [`open_with_config`](Self::open_with_config) for details on
    /// crash recovery and reconciliation.
    pub async fn open(path: &Path) -> Result<Self, crate::Error> {
        Self::open_with_config(path, Config::default()).await
    }

    /// Append a fully-formed entry to the store: log write, manifest bump,
    /// index add, and cache push.
    ///
    /// Assumes the caller already holds the exclusive file lock and has
    /// refreshed state via `refresh_if_stale`. Lock order is writer →
    /// manifest → index → pending_index_writes → cache, matching the
    /// documented ordering.
    fn append_entry_to_store(&self, entry: MemoryEntry) -> Result<(), crate::Error> {
        let checksum = entry.checksum;

        {
            let mut writer = lock_mutex(&self.writer)?;
            let w = writer.as_mut().ok_or_else(|| {
                crate::Error::Io(std::io::Error::other("Log writer not available"))
            })?;
            w.append(&entry)?;
        }

        {
            let mut manifest = lock_mutex(&self.manifest)?;
            manifest.entry_count += 1;
            manifest.last_checksum = checksum;
            manifest.oldest_timestamp = Some(match manifest.oldest_timestamp {
                Some(current) => current.min(entry.timestamp),
                None => entry.timestamp,
            });
            manifest.newest_timestamp = Some(match manifest.newest_timestamp {
                Some(current) => current.max(entry.timestamp),
                None => entry.timestamp,
            });
            manifest.updated_at = Self::now_millis();
            manifest.save(&self.base_path)?;
        }

        {
            let mut index = lock_mutex(&self.index)?;
            index.add_entry(&entry)?;
            let mut pending = lock_mutex(&self.pending_index_writes)?;
            *pending += 1;
            if *pending >= INDEX_COMMIT_BATCH_SIZE {
                index.commit()?;
                *pending = 0;
            }
        }

        {
            let mut cache = lock_mutex(&self.cache)?;
            cache.push(entry);
        }

        Ok(())
    }

    /// Store a new memory entry and return its unique ID.
    ///
    /// The entry is appended to the binary log, indexed for full-text search,
    /// and (if the `model2vec` feature is enabled) embedded for semantic
    /// search. The returned ID is a UUID v4 string that can be used with
    /// [`get`](Self::get) to retrieve the entry later.
    ///
    /// If [`Config::max_entries`] is set and the store reaches the limit
    /// after this write, the oldest entries are automatically rotated out.
    ///
    /// # Arguments
    ///
    /// * `agent_name` — name of the agent storing the memory
    /// * `entry_type` — category tag for the memory (e.g. [`EntryType::Discovery`])
    /// * `summary` — short, human-readable summary
    /// * `content` — full content of the memory
    ///
    /// # Errors
    ///
    /// Returns an error if the log cannot be written, the index cannot be
    /// updated, or the embedding model fails.
    pub async fn remember(
        &self,
        agent_name: &str,
        entry_type: EntryType,
        summary: &str,
        content: &str,
    ) -> Result<String, crate::Error> {
        let _guard = self.file_lock.lock_exclusive()?;
        self.refresh_if_stale()?;

        let check_rotation = self.config.max_entries.is_some();

        let prev_checksum = {
            let manifest = lock_mutex(&self.manifest)?;
            manifest.last_checksum
        };

        let entry = MemoryEntry::new(
            agent_name.to_string(),
            entry_type,
            summary.to_string(),
            content.to_string(),
            prev_checksum,
        );

        let entry_id = entry.id.clone();

        let entry_to_write = if self.embeddings.is_available() {
            let mut entry_with_embedding = entry.clone();
            match self.embeddings.embed(content) {
                Ok(embedding) => {
                    entry_with_embedding.embedding = Some(embedding);
                    entry_with_embedding.checksum = MemoryEntry::compute_checksum(
                        &entry_with_embedding.id,
                        &entry_with_embedding.agent_name,
                        entry_with_embedding.entry_type,
                        &entry_with_embedding.summary,
                        &entry_with_embedding.content,
                        entry_with_embedding.timestamp,
                        entry_with_embedding.prev_checksum,
                        entry_with_embedding.embedding.as_deref(),
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to compute embedding for entry {}: {}", entry_id, e);
                }
            }
            entry_with_embedding
        } else {
            entry
        };

        self.append_entry_to_store(entry_to_write)?;

        if check_rotation {
            let should_rotate = {
                let manifest = lock_mutex(&self.manifest)?;
                if let Some(max_entries) = self.config.max_entries {
                    manifest.entry_count >= max_entries
                } else {
                    false
                }
            };

            if should_rotate {
                self.rotate_old_entries().await?;
            }
        }

        self.update_fingerprint()?;
        Ok(entry_id)
    }

    async fn rotate_old_entries(&self) -> Result<(), crate::Error> {
        let max_entries = self.config.max_entries.unwrap_or(u64::MAX);

        let entries = {
            let cache = lock_mutex(&self.cache)?;
            cache.ordered.clone()
        };

        if entries.len() as u64 <= max_entries {
            return Ok(());
        }

        let entries_to_remove = entries.len() as u64 - max_entries;
        let entries_to_keep: Vec<MemoryEntry> = entries
            .into_iter()
            .skip(entries_to_remove as usize)
            .collect();

        let mut new_prev_checksum = 0u32;
        let mut relinked_entries = Vec::with_capacity(entries_to_keep.len());
        for entry in &entries_to_keep {
            let mut entry_with_prev = entry.clone();
            entry_with_prev.prev_checksum = new_prev_checksum;
            entry_with_prev.checksum = MemoryEntry::compute_checksum(
                &entry_with_prev.id,
                &entry_with_prev.agent_name,
                entry_with_prev.entry_type,
                &entry_with_prev.summary,
                &entry_with_prev.content,
                entry_with_prev.timestamp,
                entry_with_prev.prev_checksum,
                entry_with_prev.embedding.as_deref(),
            );
            new_prev_checksum = entry_with_prev.checksum;
            relinked_entries.push(entry_with_prev);
        }

        self.rewrite_log_atomically(&relinked_entries)?;

        let mut index = lock_mutex(&self.index)?;
        index.clear()?;
        index.rebuild_from_entries(&relinked_entries)?;
        drop(index);
        self.reset_pending_index_writes()?;

        let mut manifest = lock_mutex(&self.manifest)?;
        manifest.entry_count = relinked_entries.len() as u64;
        manifest.last_checksum = new_prev_checksum;
        manifest.oldest_timestamp = relinked_entries.first().map(|e| e.timestamp);
        manifest.newest_timestamp = relinked_entries.last().map(|e| e.timestamp);
        manifest.updated_at = Self::now_millis();
        manifest.save(&self.base_path)?;

        let mut cache = lock_mutex(&self.cache)?;
        cache.replace(relinked_entries);

        Ok(())
    }

    /// Search memories using hybrid BM25 + semantic search.
    ///
    /// When the `model2vec` feature is enabled, the query is embedded and
    /// results are ranked using Reciprocal Rank Fusion (RRF) across both
    /// BM25 keyword scores and cosine similarity scores. When embeddings
    /// are unavailable, only BM25 keyword search is used.
    ///
    /// Raw retrieval scores are then blended with an entry-type importance
    /// weight and an exponential recency decay (120-day half-life, 0.6
    /// floor), and results whose adjusted score falls below 20% of the best
    /// adjusted score are dropped. Results are returned in descending
    /// adjusted-score order, limited to at most `limit` entries. When
    /// `agent_name` is `Some`, only entries created by the given agent
    /// are returned.
    ///
    /// # Arguments
    ///
    /// * `query` — natural language search query
    /// * `limit` — maximum number of results to return
    /// * `agent_name` — optional agent name to filter results by
    pub async fn search_memory(
        &self,
        query: &str,
        limit: usize,
        agent_name: Option<&str>,
    ) -> Result<Vec<SearchResult>, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        self.commit_pending_index_writes(true)?;
        let index = lock_mutex(&self.index)?;

        let query_embedding = if self.embeddings.is_available() {
            self.embeddings.embed(query).ok()
        } else {
            None
        };

        // Always request more candidates than the caller asked for: results are
        // post-filtered by agent and by supersession, and the extra headroom
        // keeps those filters from starving the final result set.
        let fetch_limit = limit.saturating_mul(4).max(limit + 16);

        let search_results = if let Some(ref emb) = query_embedding {
            index.hybrid_search(query, Some(emb), fetch_limit)?
        } else {
            index.search(query, fetch_limit)?
        };

        let cache = lock_mutex(&self.cache)?;
        let usage = lock_mutex(&self.usage_counts)?;

        // Entries superseded by tombstones are hidden from results, as are
        // the tombstones themselves.
        let superseded = superseded_ids(&cache);

        let now_ms = Self::now_millis();

        // First pass: filter candidates and blend each raw retrieval score
        // with entry-type importance, exponential recency decay, and
        // recorded usage.
        let mut candidates: Vec<SearchResult> = Vec::new();
        for (id, score) in search_results {
            if superseded.contains(&id.to_ascii_lowercase()) {
                continue;
            }
            if let Some(entry) = cache.by_id.get(&id) {
                if entry.summary.starts_with(SUPERSEDES_PREFIX) {
                    continue;
                }
                if let Some(filter_name) = agent_name
                    && entry.agent_name != filter_name
                {
                    continue;
                }
                let use_count = usage.get(&id).copied().unwrap_or(0);
                candidates.push(SearchResult {
                    id: id.clone(),
                    entry: entry.clone(),
                    score: adjust_score(score, entry, now_ms, use_count),
                });
            }
        }

        // Relative score floor: drop weak tail matches below a fixed
        // fraction of the best adjusted score.
        if let Some(best) = candidates
            .iter()
            .map(|r| r.score)
            .fold(None, |acc: Option<f32>, s| {
                Some(acc.map_or(s, |a| a.max(s)))
            })
        {
            let cutoff = best * MIN_SCORE_RATIO;
            candidates.retain(|r| r.score >= cutoff);
        }

        // Re-sort by adjusted score (the retrieval engine's order reflects
        // only the raw fused score).
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);

        Ok(candidates)
    }

    /// Ask a natural language question and get a formatted text answer.
    ///
    /// This is a convenience wrapper around
    /// [`ask_memory_with_options`](Self::ask_memory_with_options) using
    /// [`AskOptions::default`] (200-character content previews).
    pub async fn ask_memory(
        &self,
        question: &str,
        agent_name: Option<&str>,
    ) -> Result<String, crate::Error> {
        self.ask_memory_with_options(question, agent_name, AskOptions::default())
            .await
    }

    /// Ask a natural language question and get a formatted text answer.
    ///
    /// This is a convenience wrapper around [`search_memory`](Self::search_memory)
    /// that returns the top 5 results as a human-readable string, with each
    /// entry's agent name, type, summary, and a preview of its content.
    ///
    /// `options.content_chars` controls how much of each entry's content is
    /// included: `0` includes the full content, any other value truncates
    /// at that many bytes (on a UTF-8 char boundary).
    ///
    /// When `agent_name` is `Some`, only entries from that agent are
    /// considered.
    ///
    /// Returns `"No relevant memories found."` if no matches are found.
    pub async fn ask_memory_with_options(
        &self,
        question: &str,
        agent_name: Option<&str>,
        options: AskOptions,
    ) -> Result<String, crate::Error> {
        // Note: search_memory acquires its own shared lock, so we don't double-lock here.
        let results = self.search_memory(question, 5, agent_name).await?;

        if results.is_empty() {
            return Ok("No relevant memories found.".to_string());
        }

        let mut response = String::from("Based on my memory:\n\n");
        for (i, result) in results.iter().enumerate() {
            response.push_str(&format!(
                "{}. [{}] ({}) {}\n",
                i + 1,
                result.entry.entry_type,
                result.entry.agent_name,
                result.entry.summary
            ));
            let content = if options.content_chars == 0 {
                result.entry.content.as_str()
            } else {
                truncate_at_char_boundary(&result.entry.content, options.content_chars)
            };
            if content.len() == result.entry.content.len() {
                response.push_str(&format!("   {}\n\n", content));
            } else {
                response.push_str(&format!("   {}...\n\n", content));
            }
        }

        Ok(response)
    }

    /// Record that an entry was useful (F6 usage tracking).
    ///
    /// Appends a usage event for the given entry id to the append-only
    /// usage.jsonl sidecar and bumps the in-memory count. Repeated marks
    /// accumulate: each use adds USAGE_BOOST_PER_USE to the entry's
    /// search-score multiplier, up to USAGE_COUNT_CAP uses.
    ///
    /// The entry log (log.bin) is untouched, so stores stay byte-compatible
    /// with upstream 0.3.5.
    ///
    /// Accepts a full UUID or any unambiguous id prefix. Returns the
    /// resolved entry id and the new total use count.
    ///
    /// # Errors
    ///
    /// Returns an error if the entry id (or prefix) matches no entry or is
    /// ambiguous, or if the usage sidecar cannot be appended to.
    pub async fn mark_used(&self, entry_id: &str) -> Result<(String, u32), crate::Error> {
        let _guard = self.file_lock.lock_exclusive()?;
        self.refresh_if_stale()?;

        let needle = entry_id.to_ascii_lowercase();
        let id = {
            let cache = lock_mutex(&self.cache)?;
            if cache.by_id.contains_key(&needle) {
                needle
            } else {
                // Prefix resolution: accept any unambiguous id prefix.
                let mut matches = cache.by_id.keys().filter(|id| id.starts_with(&needle));
                match (matches.next(), matches.next()) {
                    (Some(first), None) => first.clone(),
                    (Some(_), Some(_)) => {
                        return Err(crate::Error::Serialization(format!(
                            "ambiguous entry id prefix: {entry_id}"
                        )));
                    }
                    (None, _) => {
                        return Err(crate::Error::Serialization(format!(
                            "entry id not found: {entry_id}"
                        )));
                    }
                }
            }
        };

        let event = UsageEvent {
            id: id.clone(),
            timestamp: Self::now_millis(),
        };
        let mut line = serde_json::to_string(&event)?;
        line.push('\n');

        let path = usage_log_path(&self.base_path);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        file.write_all(line.as_bytes())?;
        file.sync_all()?;

        let mut usage = lock_mutex(&self.usage_counts)?;
        let count = usage.entry(id.clone()).or_insert(0);
        *count += 1;
        Ok((id, *count))
    }

    /// Find entries that mention a given entity (F7 entity index).
    ///
    /// Looks the normalized term up in the in-memory inverted entity
    /// index built from entry summaries and content. An exact token match
    /// wins; otherwise any indexed entity containing the term as a
    /// substring counts as a hit. Matching entries are returned newest
    /// first, capped at limit.
    ///
    /// # Errors
    ///
    /// Returns an error if the store lock cannot be acquired.
    pub async fn find_entities(
        &self,
        term: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        let needle = term.trim().to_ascii_lowercase();
        let cache = lock_mutex(&self.cache)?;

        let mut ids: HashSet<String> = HashSet::new();
        if let Some(exact) = cache.entities.get(&needle) {
            ids.extend(exact.iter().cloned());
        } else {
            // Fallback: substring containment, so a fragment of a path or
            // identifier (e.g. "opencode" for e:/x/opencode.json) still hits.
            for (token, entry_ids) in &cache.entities {
                if token.contains(&needle) {
                    ids.extend(entry_ids.iter().cloned());
                }
            }
        }

        let mut entries: Vec<MemoryEntry> = ids
            .iter()
            .filter_map(|id| cache.by_id.get(id).cloned())
            .collect();
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(limit);
        Ok(entries)
    }

    /// Return aggregate statistics about the memory store.
    ///
    /// The returned [`MemoryStats`] includes the total entry count, log file
    /// size in bytes, and the timestamps of the oldest and newest entries.
    pub async fn memory_stats(&self) -> Result<MemoryStats, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        let manifest = lock_mutex(&self.manifest)?;

        let log_path = Manifest::log_path(&self.base_path);
        let file_size = log_reader::file_size(&log_path)?;

        Ok(MemoryStats {
            total_entries: manifest.entry_count,
            file_size_bytes: file_size,
            oldest_timestamp: manifest.oldest_timestamp,
            newest_timestamp: manifest.newest_timestamp,
        })
    }

    /// Retrieve entries in chronological order with optional filtering.
    ///
    /// Returns entries sorted by timestamp. Use [`TimelineOptions`] to
    /// control the direction (`reverse`), limit, and time-range filters
    /// (`since` / `until`).
    pub async fn timeline(
        &self,
        options: TimelineOptions,
    ) -> Result<Vec<MemoryEntry>, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        let cache = lock_mutex(&self.cache)?;
        let mut entries = Vec::with_capacity(options.limit);

        let matches_filters = |entry: &MemoryEntry| {
            let since_ok = options.since.is_none_or(|s| entry.timestamp >= s);
            let until_ok = options.until.is_none_or(|u| entry.timestamp <= u);
            let agent_ok = options
                .agent_name
                .as_ref()
                .is_none_or(|name| entry.agent_name == *name);
            since_ok && until_ok && agent_ok
        };

        if options.reverse {
            for entry in cache.ordered.iter().rev() {
                if matches_filters(entry) {
                    entries.push(entry.clone());
                    if entries.len() == options.limit {
                        break;
                    }
                }
            }
        } else {
            for entry in &cache.ordered {
                if matches_filters(entry) {
                    entries.push(entry.clone());
                    if entries.len() == options.limit {
                        break;
                    }
                }
            }
        }

        Ok(entries)
    }

    /// Validate the CRC32 checksum chain of the binary log.
    ///
    /// Reads every entry in the log and verifies that each entry's checksum
    /// matches its computed value and that each entry's `prev_checksum`
    /// matches the preceding entry's checksum. Returns `true` if the chain
    /// is intact, `false` if any entry fails verification.
    pub async fn verify(&self) -> Result<bool, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;

        let log_path = Manifest::log_path(&self.base_path);
        log_reader::validate_checksum_chain(&log_path)
    }

    /// Drop and rebuild the full-text search index from the in-memory cache.
    ///
    /// This is useful if the index has become inconsistent (e.g. after a
    /// crash during a batch write). The index is rebuilt from the cached
    /// entries, so no disk I/O on the log file is required.
    pub async fn rebuild_index(&self) -> Result<(), crate::Error> {
        let _guard = self.file_lock.lock_exclusive()?;
        self.refresh_if_stale()?;

        let entries = {
            let cache = lock_mutex(&self.cache)?;
            cache.ordered.clone()
        };

        let mut index = lock_mutex(&self.index)?;
        index.rebuild_from_entries(&entries)?;
        drop(index);
        self.reset_pending_index_writes()?;

        Ok(())
    }

    /// Retrieve a single memory entry by its full UUID string.
    ///
    /// Returns `Ok(None)` if no entry with the given ID exists.
    /// Lookups are O(1) via an in-memory hash map.
    pub async fn get(&self, id: &str) -> Result<Option<MemoryEntry>, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        let cache = lock_mutex(&self.cache)?;
        Ok(cache.by_id.get(id).cloned())
    }

    /// Resolve a full UUID or an unambiguous UUID prefix to one entry.
    ///
    /// Returns `Ok(None)` for no match and a serialization error when a
    /// prefix matches more than one entry.
    pub async fn get_by_id_or_prefix(
        &self,
        id_or_prefix: &str,
    ) -> Result<Option<MemoryEntry>, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        let needle = id_or_prefix.to_ascii_lowercase();
        let cache = lock_mutex(&self.cache)?;
        if let Some(entry) = cache.by_id.get(&needle) {
            return Ok(Some(entry.clone()));
        }

        let mut matches = cache
            .by_id
            .values()
            .filter(|entry| entry.id.to_ascii_lowercase().starts_with(&needle));
        let first = matches.next().cloned();
        if matches.next().is_some() {
            return Err(crate::Error::Serialization(format!(
                "Entry id prefix is ambiguous: {id_or_prefix}"
            )));
        }
        Ok(first)
    }

    /// Remove entries with invalid checksums and rewrite the log atomically.
    ///
    /// Each entry's checksum is recomputed and compared to the stored value.
    /// Entries that fail verification are discarded. The remaining entries
    /// are re-linked into a new checksum chain and written to a temporary
    /// file, which is then atomically renamed over the original log.
    ///
    /// The manifest, index, and in-memory cache are all updated to reflect
    /// the compacted state.
    pub async fn compact(&self) -> Result<(), crate::Error> {
        self.compact_with_options(CompactOptions::default()).await?;
        Ok(())
    }

    /// Compact the memory store with explicit options.
    ///
    /// Behaves like [`compact`](Self::compact) and additionally — when
    /// [`CompactOptions::prune_superseded`] is set — physically removes
    /// entries hidden by `SUPERSEDES` tombstones along with the tombstones
    /// themselves. Returns a [`CompactReport`] describing what changed.
    pub async fn compact_with_options(
        &self,
        options: CompactOptions,
    ) -> Result<CompactReport, crate::Error> {
        let _guard = self.file_lock.lock_exclusive()?;
        self.refresh_if_stale()?;

        let entries = {
            let cache = lock_mutex(&self.cache)?;
            cache.ordered.clone()
        };

        let entries_before = entries.len() as u64;

        // Optionally drop entries that have been superseded by tombstones,
        // along with the tombstones themselves. The checksum chain is
        // relinked below, so removing entries here is safe.
        let entries: Vec<MemoryEntry> = if options.prune_superseded {
            let superseded: HashSet<String> = entries
                .iter()
                .filter_map(|e| parse_superseded_id(&e.summary))
                .collect();
            entries
                .into_iter()
                .filter(|e| {
                    !superseded.contains(&e.id.to_ascii_lowercase())
                        && !e.summary.starts_with(SUPERSEDES_PREFIX)
                })
                .collect()
        } else {
            entries
        };
        let pruned_superseded = entries_before - entries.len() as u64;

        let valid_entries: Vec<MemoryEntry> = entries
            .into_iter()
            .filter(|e| {
                let expected = MemoryEntry::compute_checksum(
                    &e.id,
                    &e.agent_name,
                    e.entry_type,
                    &e.summary,
                    &e.content,
                    e.timestamp,
                    e.prev_checksum,
                    e.embedding.as_deref(),
                );
                e.checksum == expected
            })
            .collect();

        // Re-link the checksum chain so that each entry's prev_checksum
        // points to the actual preceding entry (entries may have been
        // removed by the filter above, breaking the original chain).
        let mut new_prev_checksum = 0u32;
        let mut relinked_entries = Vec::with_capacity(valid_entries.len());
        for entry in &valid_entries {
            let mut relinked = entry.clone();
            relinked.prev_checksum = new_prev_checksum;
            relinked.checksum = MemoryEntry::compute_checksum(
                &relinked.id,
                &relinked.agent_name,
                relinked.entry_type,
                &relinked.summary,
                &relinked.content,
                relinked.timestamp,
                relinked.prev_checksum,
                relinked.embedding.as_deref(),
            );
            new_prev_checksum = relinked.checksum;
            relinked_entries.push(relinked);
        }

        let entries_after = relinked_entries.len() as u64;

        let mut index = lock_mutex(&self.index)?;

        self.rewrite_log_atomically(&relinked_entries)?;

        index.rebuild_from_entries(&relinked_entries)?;
        drop(index);
        self.reset_pending_index_writes()?;

        {
            let mut manifest = lock_mutex(&self.manifest)?;
            manifest.entry_count = relinked_entries.len() as u64;
            manifest.last_checksum = relinked_entries.last().map_or(0, |e| e.checksum);
            manifest.oldest_timestamp = relinked_entries.first().map(|e| e.timestamp);
            manifest.newest_timestamp = relinked_entries.last().map(|e| e.timestamp);
            manifest.updated_at = Self::now_millis();
            manifest.save(&self.base_path)?;
        }

        {
            let mut cache = lock_mutex(&self.cache)?;
            cache.replace(relinked_entries);
        }

        self.update_fingerprint()?;
        Ok(CompactReport {
            entries_before,
            entries_after,
            pruned_superseded,
        })
    }

    /// Consolidate near-duplicate entries using their stored embeddings (F8).
    ///
    /// Entries are clustered by pairwise cosine similarity computed from the
    /// embeddings stored at write time. Each cluster is anchored on its newest
    /// member; any other member whose similarity to that anchor meets
    /// `similarity_threshold` joins the group (greedy leader clustering, so
    /// there is no transitive chaining between merely related entries). For
    /// every cluster of at least `min_group_size` members, the newest entry is
    /// kept and the older members are superseded by writing `SUPERSEDES`
    /// tombstones.
    ///
    /// The append-only log is preserved; run [`compact_with_options`](Self::compact_with_options)
    /// with [`CompactOptions::prune_superseded`] set afterwards to physically
    /// remove the superseded entries and tombstones.
    ///
    /// # Parameters
    ///
    /// * `similarity_threshold` — minimum cosine similarity for two entries
    ///   to be treated as duplicates (0.0..=1.0, higher is stricter).
    /// * `min_group_size` — minimum cluster size before it is consolidated.
    ///   Use 2 to merge duplicate pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the store lock cannot be acquired or a write
    /// fails. Entries without stored embeddings are never clustered.
    pub async fn consolidate(
        &self,
        similarity_threshold: f32,
        min_group_size: usize,
    ) -> Result<ConsolidateReport, crate::Error> {
        let _guard = self.file_lock.lock_exclusive()?;
        self.refresh_if_stale()?;

        let entries_before = {
            let manifest = lock_mutex(&self.manifest)?;
            manifest.entry_count
        };

        let entries = {
            let cache = lock_mutex(&self.cache)?;
            cache.ordered.clone()
        };

        // Set of ids already hidden by a tombstone; never re-cluster them.
        let superseded_ids: HashSet<String> = entries
            .iter()
            .filter_map(|e| parse_superseded_id(&e.summary))
            .collect();

        // Only entries with a stored embedding (and not already superseded
        // or a tombstone) can take part in semantic clustering.
        let candidates: Vec<MemoryEntry> = entries
            .iter()
            .filter(|e| {
                e.embedding.is_some()
                    && !e.summary.starts_with(SUPERSEDES_PREFIX)
                    && !superseded_ids.contains(&e.id.to_ascii_lowercase())
            })
            .cloned()
            .collect();

        // Greedy leader clustering: repeatedly anchor on the newest unused
        // candidate and absorb every other unused candidate close to it.
        let n = candidates.len();
        let mut kept: Vec<bool> = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        loop {
            let mut anchor: Option<usize> = None;
            for i in 0..n {
                if !kept[i]
                    && (anchor.is_none()
                        || candidates[i].timestamp > candidates[anchor.unwrap()].timestamp)
                {
                    anchor = Some(i);
                }
            }
            let Some(anchor_idx) = anchor else { break };

            kept[anchor_idx] = true;
            let mut cluster = vec![anchor_idx];
            let a_emb = candidates[anchor_idx].embedding.as_ref().unwrap();

            for j in 0..n {
                if kept[j] || j == anchor_idx {
                    continue;
                }
                let b_emb = candidates[j].embedding.as_ref().unwrap();
                let close = crate::search::compute_cosine_similarity(a_emb, b_emb)
                    .is_some_and(|sim| sim >= similarity_threshold);
                if close {
                    kept[j] = true;
                    cluster.push(j);
                }
            }

            clusters.push(cluster);
        }

        let mut superseded = 0u64;
        let mut clusters_merged = 0u64;

        for cluster in clusters {
            if cluster.len() < min_group_size {
                continue;
            }
            clusters_merged += 1;

            // cluster[0] is the newest member (the anchor). Keep it and
            // supersede every other member.
            let keep = &candidates[cluster[0]];
            for &odx in cluster.iter().skip(1) {
                let dup = &candidates[odx];
                let reason = format!("consolidated: near-duplicate of {}", keep.id);
                let prev_checksum = {
                    let manifest = lock_mutex(&self.manifest)?;
                    manifest.last_checksum
                };
                let tombstone = MemoryEntry::new(
                    dup.agent_name.clone(),
                    EntryType::Discovery,
                    format!("{SUPERSEDES_PREFIX}{} — {reason}", dup.id),
                    String::new(),
                    prev_checksum,
                );
                self.append_entry_to_store(tombstone)?;
                superseded += 1;
            }
        }

        self.update_fingerprint()?;
        Ok(ConsolidateReport {
            entries_before,
            clusters_merged,
            superseded,
        })
    }

    /// Export all memory entries to a JSON file.
    ///
    /// Writes a pretty-printed JSON array of all [`MemoryEntry`] records to
    /// the given `path`. The exported file can later be imported into
    /// another store via [`import`](Self::import).
    pub async fn export(&self, path: &Path) -> Result<(), crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        let entries = {
            let cache = lock_mutex(&self.cache)?;
            cache.ordered.clone()
        };

        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;

        std::fs::write(path, json)?;
        Ok(())
    }

    /// Import memory entries from a JSON file.
    ///
    /// Reads a JSON array of [`MemoryEntry`] records from `path` and appends
    /// them to this store. Each imported entry is re-linked into the current
    /// checksum chain (its `prev_checksum` and `checksum` fields are
    /// recomputed). Returns the number of entries imported.
    ///
    /// The JSON format is the same as produced by [`export`](Self::export).
    pub async fn import(&self, path: &Path) -> Result<u64, crate::Error> {
        let _guard = self.file_lock.lock_exclusive()?;
        self.refresh_if_stale()?;

        let content = std::fs::read_to_string(path)?;

        let entries: Vec<MemoryEntry> = serde_json::from_str(&content)
            .map_err(|e| crate::Error::Serialization(e.to_string()))?;

        let mut count = 0u64;
        let mut imported_entries = Vec::with_capacity(entries.len());
        let mut writer = lock_mutex(&self.writer)?;
        let mut manifest = lock_mutex(&self.manifest)?;
        let mut prev_checksum = manifest.last_checksum;
        let mut oldest_timestamp = manifest.oldest_timestamp;
        let mut newest_timestamp = manifest.newest_timestamp;

        let w = writer
            .as_mut()
            .ok_or_else(|| crate::Error::Io(std::io::Error::other("Log writer not available")))?;

        for entry in &entries {
            let mut relinked = entry.clone();
            relinked.prev_checksum = prev_checksum;
            relinked.checksum = MemoryEntry::compute_checksum(
                &relinked.id,
                &relinked.agent_name,
                relinked.entry_type,
                &relinked.summary,
                &relinked.content,
                relinked.timestamp,
                relinked.prev_checksum,
                relinked.embedding.as_deref(),
            );

            w.append(&relinked)?;
            prev_checksum = relinked.checksum;
            let relinked_timestamp = relinked.timestamp;

            oldest_timestamp = Some(match oldest_timestamp {
                Some(current) => current.min(relinked_timestamp),
                None => relinked_timestamp,
            });
            newest_timestamp = Some(match newest_timestamp {
                Some(current) => current.max(relinked_timestamp),
                None => relinked_timestamp,
            });

            imported_entries.push(relinked);

            count += 1;
        }

        manifest.entry_count += count;
        manifest.last_checksum = prev_checksum;
        manifest.oldest_timestamp = oldest_timestamp;
        manifest.newest_timestamp = newest_timestamp;
        manifest.updated_at = Self::now_millis();
        manifest.save(&self.base_path)?;

        drop(manifest);
        drop(writer);

        self.commit_pending_index_writes(true)?;
        let mut index = lock_mutex(&self.index)?;
        for entry in &imported_entries {
            index.add_entry(entry)?;
        }
        index.commit()?;
        drop(index);
        self.reset_pending_index_writes()?;

        let mut cache = lock_mutex(&self.cache)?;
        for entry in imported_entries {
            cache.push(entry);
        }

        self.update_fingerprint()?;
        Ok(count)
    }
}
