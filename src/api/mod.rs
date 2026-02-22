use crate::embeddings::EmbeddingBackend;
use crate::error::lock_mutex;
use crate::search::IndexManager;
use crate::storage::file_lock::FileLock;
use crate::storage::log_reader;
use crate::storage::{LogWriter, Manifest};
use crate::types::{Config, EntryType, MemoryEntry, MemoryStats, SearchResult, TimelineOptions};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::constants::LOG_FILENAME;

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
    unsafe {
        if libc::kill(pid_i32, 0) == 0 {
            return true;
        }
        // EPERM means the process exists but we can't signal it — still alive.
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}

#[cfg(windows)]
fn is_process_alive(pid: u32) -> bool {
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;

    extern "system" {
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
}

struct OpenReconciliation {
    manifest: Manifest,
    entries: Vec<MemoryEntry>,
}

impl EntryCache {
    fn from_entries(entries: Vec<MemoryEntry>) -> Self {
        let by_id = entries
            .iter()
            .map(|entry| (entry.id.clone(), entry.clone()))
            .collect();

        Self {
            ordered: entries,
            by_id,
        }
    }

    fn empty() -> Self {
        Self::from_entries(Vec::new())
    }

    fn replace(&mut self, entries: Vec<MemoryEntry>) {
        *self = Self::from_entries(entries);
    }

    fn push(&mut self, entry: MemoryEntry) {
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
        let temp_path = self
            .base_path
            .join(format!("{LOG_FILENAME}.rewrite.{}.tmp", Self::now_millis()));

        {
            let mut writer = lock_mutex(&self.writer)?;
            *writer = None;
        }

        {
            let mut temp_writer = LogWriter::new(&temp_path)?;
            for entry in rewritten_entries {
                temp_writer.append(entry)?;
            }
            temp_writer.flush()?;
        }

        std::fs::rename(&temp_path, &log_path)?;

        let dir_file = OpenOptions::new().read(true).open(&self.base_path)?;
        dir_file.sync_all()?;

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

        let checksum = entry_to_write.checksum;

        {
            let mut writer = lock_mutex(&self.writer)?;
            let w = writer.as_mut().ok_or_else(|| {
                crate::Error::Io(std::io::Error::other("Log writer not available"))
            })?;
            w.append(&entry_to_write)?;
        }

        {
            let mut manifest = lock_mutex(&self.manifest)?;
            manifest.entry_count += 1;
            manifest.last_checksum = checksum;
            manifest.oldest_timestamp = Some(match manifest.oldest_timestamp {
                Some(current) => current.min(entry_to_write.timestamp),
                None => entry_to_write.timestamp,
            });
            manifest.newest_timestamp = Some(match manifest.newest_timestamp {
                Some(current) => current.max(entry_to_write.timestamp),
                None => entry_to_write.timestamp,
            });
            manifest.updated_at = Self::now_millis();
            manifest.save(&self.base_path)?;
        }

        {
            let mut index = lock_mutex(&self.index)?;
            index.add_entry(&entry_to_write)?;
            let mut pending = lock_mutex(&self.pending_index_writes)?;
            *pending += 1;
            if *pending >= INDEX_COMMIT_BATCH_SIZE {
                index.commit()?;
                *pending = 0;
            }
        }

        {
            let mut cache = lock_mutex(&self.cache)?;
            cache.push(entry_to_write.clone());
        }

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
    /// Results are returned in descending relevance order, limited to at
    /// most `limit` entries. When `agent_name` is `Some`, only entries
    /// created by the given agent are returned.
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

        // When filtering by agent, request more candidates to compensate for
        // entries that will be discarded after the post-filter step.
        let fetch_limit = if agent_name.is_some() {
            limit * 4
        } else {
            limit
        };

        let search_results = if let Some(ref emb) = query_embedding {
            index.hybrid_search(query, Some(emb), fetch_limit)?
        } else {
            index.search(query, fetch_limit)?
        };

        let cache = lock_mutex(&self.cache)?;

        let mut results = Vec::new();
        for (id, score) in search_results {
            if results.len() >= limit {
                break;
            }
            if let Some(entry) = cache.by_id.get(&id) {
                if let Some(filter_name) = agent_name
                    && entry.agent_name != filter_name
                {
                    continue;
                }
                results.push(SearchResult {
                    id: id.clone(),
                    entry: entry.clone(),
                    score,
                });
            }
        }

        Ok(results)
    }

    /// Ask a natural language question and get a formatted text answer.
    ///
    /// This is a convenience wrapper around [`search_memory`](Self::search_memory)
    /// that returns the top 5 results as a human-readable string, with each
    /// entry's agent name, type, summary, and a truncated preview of its
    /// content.
    ///
    /// When `agent_name` is `Some`, only entries from that agent are
    /// considered.
    ///
    /// Returns `"No relevant memories found."` if no matches are found.
    pub async fn ask_memory(
        &self,
        question: &str,
        agent_name: Option<&str>,
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
            let truncated = truncate_at_char_boundary(&result.entry.content, 200);
            if truncated.len() == result.entry.content.len() {
                response.push_str(&format!("   {}\n\n", result.entry.content));
            } else {
                response.push_str(&format!("   {}...\n\n", truncated));
            }
        }

        Ok(response)
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

    /// Retrieve a single memory entry by its UUID string.
    ///
    /// Returns `Ok(None)` if no entry with the given ID exists.
    /// Lookups are O(1) via an in-memory hash map.
    pub async fn get(&self, id: &str) -> Result<Option<MemoryEntry>, crate::Error> {
        let _guard = self.file_lock.lock_shared()?;
        self.refresh_if_stale()?;

        let cache = lock_mutex(&self.cache)?;
        Ok(cache.by_id.get(id).cloned())
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
        let _guard = self.file_lock.lock_exclusive()?;
        self.refresh_if_stale()?;

        let entries = {
            let cache = lock_mutex(&self.cache)?;
            cache.ordered.clone()
        };

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
        Ok(())
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
