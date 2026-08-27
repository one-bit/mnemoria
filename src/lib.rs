//! # Mnemoria
//!
//! Persistent, git-friendly memory storage for AI agents with hybrid
//! semantic + full-text search.
//!
//! Mnemoria provides a single-file, append-only memory store that AI assistants
//! (Claude, GPT, Cursor, or any LLM-based tool) can use to remember information
//! across conversations and sessions. Memories are stored in a binary log with
//! CRC32 checksum chaining for corruption detection and crash recovery.
//!
//! ## Key features
//!
//! - **Hybrid search** — combines BM25 full-text search (via [Tantivy]) with
//!   semantic vector search (via [model2vec]) using Reciprocal Rank Fusion.
//! - **Git-friendly** — the append-only binary format produces clean diffs and
//!   merges well in version control.
//! - **Corruption-resistant** — CRC32 checksum chain with automatic crash
//!   recovery and log truncation on open.
//! - **Multi-process safe** — advisory file locking and per-process ephemeral
//!   search indexes allow concurrent readers and writers.
//!
//! [Tantivy]: https://docs.rs/tantivy
//! [model2vec]: https://docs.rs/model2vec
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use mnemoria::{Mnemoria, EntryType};
//! use std::path::Path;
//!
//! # async fn example() -> Result<(), mnemoria::Error> {
//! // Create a new memory store
//! let memory = Mnemoria::create(Path::new("./my-memories")).await?;
//!
//! // Store a memory
//! let id = memory.remember(
//!     "my-agent",
//!     EntryType::Discovery,
//!     "Rust async patterns",
//!     "Use tokio::spawn for CPU-bound work inside async contexts",
//! ).await?;
//!
//! // Search by meaning
//! let results = memory.search_memory("async concurrency", 5, None).await?;
//! for result in &results {
//!     println!("[{}] {} (score: {:.3})", result.entry.entry_type, result.entry.summary, result.score);
//! }
//!
//! // Retrieve by ID
//! let entry = memory.get(&id).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Storage format
//!
//! A memory store is a directory containing:
//!
//! | File              | Purpose                          |
//! |-------------------|----------------------------------|
//! | `log.bin`         | Append-only binary log (rkyv)    |
//! | `manifest.json`   | Metadata and checksum state      |
//! | `mnemoria.lock`   | Advisory file lock               |
//!
//! The search index is ephemeral — rebuilt from `log.bin` on each open — and
//! is stored in the OS temp directory, not in the memory store itself.
//!
//! ## Feature flags
//!
//! | Flag       | Default | Description                                    |
//! |------------|---------|------------------------------------------------|
//! | `model2vec` | **yes** | Enables semantic embeddings via model2vec.    |
//!
//! Without `model2vec`, only BM25 keyword search is available.

pub mod api;
pub mod constants;
pub mod embeddings;
pub mod error;
pub mod search;
pub mod storage;
pub mod types;

pub use api::Mnemoria;
pub use constants::{APP_NAME, DEFAULT_MODEL_ID};
pub use error::{Error, Result, lock_mutex};
pub use types::{
    AskOptions, CompactOptions, CompactReport, Config, ConsolidateReport, DurabilityMode,
    EntryType, MemoryEntry, MemoryStats, SearchResult, TimelineOptions,
};

#[cfg(test)]
mod tests {
    use super::{
        AskOptions, CompactOptions, Config, DurabilityMode, EntryType, Mnemoria, TimelineOptions,
    };
    use crate::storage::{LogWriter, Manifest};
    use crate::types::MemoryEntry;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_temp_dir() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[tokio::test]
    async fn test_create_and_open() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        drop(memory);

        let memory = Mnemoria::open(&path).await.unwrap();
        let stats = memory.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 0);
    }

    #[tokio::test]
    async fn test_remember_entry() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        let id = memory
            .remember(
                "test-agent",
                EntryType::Decision,
                "Test decision",
                "This is a test decision content",
            )
            .await
            .unwrap();

        assert!(!id.is_empty());

        let stats = memory.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 1);

        // Verify agent_name is stored
        let entry = memory.get(&id).await.unwrap().unwrap();
        assert_eq!(entry.agent_name, "test-agent");
    }

    #[tokio::test]
    async fn test_search_memory() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember(
                "agent-a",
                EntryType::Decision,
                "User prefers dark mode",
                "The user prefers dark mode in their IDE for reduced eye strain.",
            )
            .await
            .unwrap();

        memory
            .remember(
                "agent-b",
                EntryType::Feature,
                "User likes autocomplete",
                "The user heavily relies on AI autocomplete for coding efficiency.",
            )
            .await
            .unwrap();

        let results = memory
            .search_memory("dark mode preference", 5, None)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.0);
    }

    #[tokio::test]
    async fn test_search_memory_filter_by_agent() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember(
                "agent-a",
                EntryType::Decision,
                "Dark mode setting",
                "User prefers dark mode in their IDE.",
            )
            .await
            .unwrap();

        memory
            .remember(
                "agent-b",
                EntryType::Decision,
                "Dark theme preference",
                "Dark theme is preferred for the web app.",
            )
            .await
            .unwrap();

        // Search with agent filter should only return agent-a's entry
        let results = memory
            .search_memory("dark mode", 5, Some("agent-a"))
            .await
            .unwrap();
        assert!(!results.is_empty());
        for result in &results {
            assert_eq!(result.entry.agent_name, "agent-a");
        }

        // Search without filter returns both
        let all_results = memory.search_memory("dark", 5, None).await.unwrap();
        assert!(all_results.len() >= 2);
    }

    #[tokio::test]
    async fn test_timeline() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember("test-agent", EntryType::Discovery, "First", "First entry")
            .await
            .unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Second", "Second entry")
            .await
            .unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Third", "Third entry")
            .await
            .unwrap();

        let timeline = memory
            .timeline(TimelineOptions {
                limit: 10,
                since: None,
                until: None,
                reverse: true,
                agent_name: None,
            })
            .await
            .unwrap();

        assert_eq!(timeline.len(), 3);
    }

    #[tokio::test]
    async fn test_timeline_filter_by_agent() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember("agent-a", EntryType::Discovery, "A entry", "From agent A")
            .await
            .unwrap();
        memory
            .remember("agent-b", EntryType::Discovery, "B entry", "From agent B")
            .await
            .unwrap();
        memory
            .remember(
                "agent-a",
                EntryType::Discovery,
                "A entry 2",
                "From agent A again",
            )
            .await
            .unwrap();

        let timeline = memory
            .timeline(TimelineOptions {
                limit: 10,
                since: None,
                until: None,
                reverse: true,
                agent_name: Some("agent-a".to_string()),
            })
            .await
            .unwrap();

        assert_eq!(timeline.len(), 2);
        for entry in &timeline {
            assert_eq!(entry.agent_name, "agent-a");
        }
    }

    #[tokio::test]
    async fn test_verify_checksums() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember("test-agent", EntryType::Decision, "Test", "Content")
            .await
            .unwrap();

        let valid = memory.verify().await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember("test-agent", EntryType::Discovery, "Test 1", "Content 1")
            .await
            .unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Test 2", "Content 2")
            .await
            .unwrap();

        let stats = memory.memory_stats().await.unwrap();

        assert_eq!(stats.total_entries, 2);
        assert!(stats.file_size_bytes > 0);
        assert!(stats.oldest_timestamp.is_some());
        assert!(stats.newest_timestamp.is_some());
    }

    #[tokio::test]
    async fn test_export_import() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember("test-agent", EntryType::Decision, "Test", "Content")
            .await
            .unwrap();

        let export_path = temp_dir.path().join("export.json");
        memory.export(&export_path).await.unwrap();

        let mem2_path = temp_dir.path().join("mem2");
        let memory2 = Mnemoria::create(&mem2_path).await.unwrap();
        let count = memory2.import(&export_path).await.unwrap();

        assert_eq!(count, 1);

        // Verify agent_name survives export/import
        let entries = memory2.timeline(TimelineOptions::default()).await.unwrap();
        assert_eq!(entries[0].agent_name, "test-agent");
    }

    #[tokio::test]
    async fn test_get_entry() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        let id = memory
            .remember("test-agent", EntryType::Decision, "Test", "Content")
            .await
            .unwrap();

        let entry = memory.get(&id).await.unwrap();
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.summary, "Test");
        assert_eq!(entry.agent_name, "test-agent");
    }

    #[tokio::test]
    async fn test_compact() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember("test-agent", EntryType::Discovery, "Test 1", "Content 1")
            .await
            .unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Test 2", "Content 2")
            .await
            .unwrap();

        memory.compact().await.unwrap();

        let stats = memory.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 2);
    }

    #[tokio::test]
    async fn test_rebuild_index() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember("test-agent", EntryType::Discovery, "Test", "Content")
            .await
            .unwrap();

        memory.rebuild_index().await.unwrap();

        let results = memory.search_memory("Test", 5, None).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_config_with_max_entries() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let config = Config {
            max_entries: Some(2),
            ..Config::default()
        };

        let memory = Mnemoria::create_with_config(&path, config).await.unwrap();

        memory
            .remember("test-agent", EntryType::Discovery, "Entry 1", "Content 1")
            .await
            .unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Entry 2", "Content 2")
            .await
            .unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Entry 3", "Content 3")
            .await
            .unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Entry 4", "Content 4")
            .await
            .unwrap();

        let stats = memory.memory_stats().await.unwrap();
        assert!(stats.total_entries <= 2);
    }

    #[tokio::test]
    async fn test_open_recovers_when_manifest_update_was_missed() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Entry 1", "Content 1")
            .await
            .unwrap();

        let manifest_before = Manifest::load(&path).unwrap();
        let entry2 = MemoryEntry::new(
            "test-agent".to_string(),
            EntryType::Discovery,
            "Entry 2".to_string(),
            "Content 2".to_string(),
            manifest_before.last_checksum,
        );

        let log_path = Manifest::log_path(&path);
        let mut writer = LogWriter::new(&log_path).unwrap();
        writer.append(&entry2).unwrap();
        drop(writer);
        drop(memory);

        let reopened = Mnemoria::open(&path).await.unwrap();
        let stats = reopened.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 2);
        assert!(reopened.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_open_truncates_partial_tail_record() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Entry 1", "Content 1")
            .await
            .unwrap();

        let log_path = Manifest::log_path(&path);
        let valid_size = std::fs::metadata(&log_path).unwrap().len();

        let mut file = OpenOptions::new().append(true).open(&log_path).unwrap();
        file.write_all(&[1, 2, 3, 4, 5, 6]).unwrap();
        file.sync_all().unwrap();
        drop(file);
        drop(memory);

        let reopened = Mnemoria::open(&path).await.unwrap();
        let stats = reopened.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.file_size_bytes, valid_size);
        assert!(reopened.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_open_recovers_from_corrupt_manifest_json() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        memory
            .remember("test-agent", EntryType::Discovery, "Entry 1", "Content 1")
            .await
            .unwrap();
        drop(memory);

        let manifest_path = Manifest::path(&path);
        std::fs::write(&manifest_path, "{not-valid-json").unwrap();

        let reopened = Mnemoria::open(&path).await.unwrap();
        let stats = reopened.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 1);
        assert!(reopened.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_semantic_hybrid_search_parity_after_reopen() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Neural memory retrieval",
                "Vector embeddings help retrieve semantically related memories.",
            )
            .await
            .unwrap();
        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "CLI ergonomics",
                "Improve command ergonomics and shell UX.",
            )
            .await
            .unwrap();

        let query = "semantic retrieval with embeddings";
        let before = memory.search_memory(query, 5, None).await.unwrap();

        if before.is_empty() {
            return;
        }

        let before_ids: Vec<String> = before.iter().map(|r| r.id.clone()).collect();
        drop(memory);

        let reopened = Mnemoria::open(&path).await.unwrap();
        let after = reopened.search_memory(query, 5, None).await.unwrap();
        let after_ids: Vec<String> = after.iter().map(|r| r.id.clone()).collect();

        assert_eq!(after_ids, before_ids);
    }

    #[tokio::test]
    async fn test_open_rebuilds_index_from_log() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Index rebuild candidate",
                "This entry should remain searchable after reopen.",
            )
            .await
            .unwrap();
        drop(memory);

        // Each open() creates a fresh per-process index from the log,
        // so entries are always searchable after reopen.
        let reopened = Mnemoria::open(&path).await.unwrap();
        let results = reopened
            .search_memory("searchable after reopen", 5, None)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_search_recovers_after_reopen_with_deferred_index_commits() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();
        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Deferred index entry",
                "This entry is written before any explicit search commit.",
            )
            .await
            .unwrap();
        drop(memory);

        let reopened = Mnemoria::open(&path).await.unwrap();
        let results = reopened
            .search_memory("deferred explicit search", 5, None)
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_two_instances_see_each_others_writes() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let instance_a = Mnemoria::create(&path).await.unwrap();
        instance_a
            .remember(
                "agent-a",
                EntryType::Discovery,
                "From A",
                "Written by instance A",
            )
            .await
            .unwrap();

        // Open a second instance on the same directory
        let instance_b = Mnemoria::open(&path).await.unwrap();
        let stats_b = instance_b.memory_stats().await.unwrap();
        assert_eq!(stats_b.total_entries, 1, "B should see A's entry on open");

        // A writes another entry
        instance_a
            .remember(
                "agent-a",
                EntryType::Decision,
                "From A again",
                "Second write by A",
            )
            .await
            .unwrap();

        // B should detect the change via refresh_if_stale
        let stats_b = instance_b.memory_stats().await.unwrap();
        assert_eq!(
            stats_b.total_entries, 2,
            "B should see A's second entry via cache invalidation"
        );

        // B writes an entry
        instance_b
            .remember(
                "agent-b",
                EntryType::Problem,
                "From B",
                "Written by instance B",
            )
            .await
            .unwrap();

        // A should see it
        let stats_a = instance_a.memory_stats().await.unwrap();
        assert_eq!(
            stats_a.total_entries, 3,
            "A should see B's entry via cache invalidation"
        );

        // Checksum chain must still be valid
        assert!(instance_a.verify().await.unwrap());
        assert!(instance_b.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_concurrent_writers_preserve_checksum_chain() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let instance_a = Mnemoria::create(&path).await.unwrap();
        let instance_b = Mnemoria::open(&path).await.unwrap();

        // Alternate writes between instances
        for i in 0..10 {
            if i % 2 == 0 {
                instance_a
                    .remember(
                        "agent-a",
                        EntryType::Discovery,
                        &format!("A-{i}"),
                        &format!("Content from A iteration {i}"),
                    )
                    .await
                    .unwrap();
            } else {
                instance_b
                    .remember(
                        "agent-b",
                        EntryType::Decision,
                        &format!("B-{i}"),
                        &format!("Content from B iteration {i}"),
                    )
                    .await
                    .unwrap();
            }
        }

        // Both should see all 10 entries
        let stats_a = instance_a.memory_stats().await.unwrap();
        let stats_b = instance_b.memory_stats().await.unwrap();
        assert_eq!(stats_a.total_entries, 10);
        assert_eq!(stats_b.total_entries, 10);

        // Checksum chain must be intact
        assert!(instance_a.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_compact_while_other_instance_open() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let instance_a = Mnemoria::create(&path).await.unwrap();
        instance_a
            .remember("agent-a", EntryType::Discovery, "Entry 1", "Content 1")
            .await
            .unwrap();
        instance_a
            .remember("agent-a", EntryType::Discovery, "Entry 2", "Content 2")
            .await
            .unwrap();

        let instance_b = Mnemoria::open(&path).await.unwrap();

        // A compacts
        instance_a.compact().await.unwrap();

        // B should still be able to read and write
        let stats_b = instance_b.memory_stats().await.unwrap();
        assert_eq!(stats_b.total_entries, 2);

        instance_b
            .remember("agent-b", EntryType::Decision, "Entry 3", "Content 3")
            .await
            .unwrap();

        let stats_a = instance_a.memory_stats().await.unwrap();
        assert_eq!(stats_a.total_entries, 3);
        assert!(instance_a.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_durability_mode_flush_only() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let config = Config {
            durability: DurabilityMode::FlushOnly,
            ..Config::default()
        };

        let memory = Mnemoria::create_with_config(&path, config).await.unwrap();

        for i in 0..10 {
            memory
                .remember(
                    "test-agent",
                    EntryType::Discovery,
                    &format!("Entry {i}"),
                    &format!("Content {i}"),
                )
                .await
                .unwrap();
        }

        let stats = memory.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 10);
        assert!(memory.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_ask_memory_with_non_ascii_content() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        // Create content with multi-byte UTF-8 characters that would panic
        // if sliced at an arbitrary byte offset.
        // Emoji (4 bytes each), CJK (3 bytes each), accented chars (2 bytes each)
        let long_content = "Hello \u{1F600}\u{1F600}\u{1F600} ".to_string()
            + &"\u{4E16}\u{754C}".repeat(50) // CJK: 世界 repeated
            + " caf\u{00E9} r\u{00E9}sum\u{00E9}"
            + &"x".repeat(200); // Ensure content > 200 bytes

        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Non-ASCII test",
                &long_content,
            )
            .await
            .unwrap();

        // This must not panic even though content > 200 bytes with multi-byte chars
        let answer = memory.ask_memory("Non-ASCII test", None).await.unwrap();
        assert!(answer.contains("...") || answer.contains("Non-ASCII"));
    }

    #[tokio::test]
    async fn test_durability_mode_none() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let config = Config {
            durability: DurabilityMode::None,
            ..Config::default()
        };

        let memory = Mnemoria::create_with_config(&path, config).await.unwrap();

        for i in 0..10 {
            memory
                .remember(
                    "test-agent",
                    EntryType::Discovery,
                    &format!("Entry {i}"),
                    &format!("Content {i}"),
                )
                .await
                .unwrap();
        }

        let stats = memory.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 10);
        assert!(memory.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_search_hides_superseded_entries_and_tombstones() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        let id_a = memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Zephyr database tuning notes",
                "The zephyr database needs connection pool tuning for throughput",
            )
            .await
            .unwrap();

        let id_b = memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Zephyr deployment checklist",
                "The zephyr deployment checklist covers rollout steps",
            )
            .await
            .unwrap();

        // Tombstone supersedes entry A (em dash separator, as produced by forget).
        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                &format!("SUPERSEDES {id_a} — replaced by newer tuning notes"),
                "obsolete",
            )
            .await
            .unwrap();

        let results = memory.search_memory("zephyr", 10, None).await.unwrap();

        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(
            ids.contains(&id_b.as_str()),
            "live entry must remain visible"
        );
        assert!(
            !ids.contains(&id_a.as_str()),
            "superseded entry must be hidden"
        );
        for r in &results {
            assert!(
                !r.entry.summary.starts_with("SUPERSEDES "),
                "tombstone must not appear in results"
            );
        }

        // ask_memory inherits the filter since it delegates to search_memory.
        let answer = memory.ask_memory("zephyr", None).await.unwrap();
        assert!(answer.contains("deployment checklist"));
        assert!(!answer.contains("connection pool"));
    }

    #[tokio::test]
    async fn test_compact_prune_superseded() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        let id_a = memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Quasar cache eviction policy",
                "The quasar cache evicts by LRU",
            )
            .await
            .unwrap();
        let id_b = memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Quasar cache eviction v2",
                "The quasar cache now evicts by LFU",
            )
            .await
            .unwrap();

        // Tombstone supersedes entry A.
        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                &format!("SUPERSEDES {id_a} — replaced by v2 policy"),
                "obsolete",
            )
            .await
            .unwrap();

        // Default compact keeps everything (byte-compatible behavior).
        let report = memory
            .compact_with_options(CompactOptions::default())
            .await
            .unwrap();
        assert_eq!(report.entries_before, 3);
        assert_eq!(report.entries_after, 3);
        assert_eq!(report.pruned_superseded, 0);

        // Pruning removes the superseded entry and its tombstone.
        let report = memory
            .compact_with_options(CompactOptions {
                prune_superseded: true,
            })
            .await
            .unwrap();
        assert_eq!(report.entries_before, 3);
        assert_eq!(report.entries_after, 1);
        assert_eq!(report.pruned_superseded, 2);

        // Only the live entry survives, and the chain still verifies.
        assert!(memory.verify().await.unwrap());
        assert!(memory.get(&id_b).await.unwrap().is_some());
        assert!(memory.get(&id_a).await.unwrap().is_none());

        // A fresh instance reading the rewritten log sees the same state.
        drop(memory);
        let reopened = Mnemoria::open(&path).await.unwrap();
        let stats = reopened.memory_stats().await.unwrap();
        assert_eq!(stats.total_entries, 1);
        assert!(reopened.verify().await.unwrap());
    }

    #[tokio::test]
    async fn test_search_recency_ranks_newer_entries_first() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        // Two entries with identical indexed text (so identical raw BM25
        // scores) but very different ages. import() preserves timestamps,
        // giving precise control over recency.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let old_ms = now_ms - 365 * 86_400_000; // one year ago

        let entries = serde_json::json!([
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "agent_name": "test-agent",
                "entry_type": "discovery",
                "summary": "Nebula retry backoff policy",
                "content": "The nebula client retries with exponential backoff",
                "embedding": null,
                "timestamp": old_ms,
                "checksum": 0,
                "prev_checksum": 0
            },
            {
                "id": "22222222-2222-4222-8222-222222222222",
                "agent_name": "test-agent",
                "entry_type": "discovery",
                "summary": "Nebula retry backoff policy",
                "content": "The nebula client retries with exponential backoff",
                "embedding": null,
                "timestamp": now_ms,
                "checksum": 0,
                "prev_checksum": 0
            }
        ]);
        let import_path = path.join("import.json");
        std::fs::write(&import_path, entries.to_string()).unwrap();
        let imported = memory.import(&import_path).await.unwrap();
        assert_eq!(imported, 2);

        // Identical raw scores mean the recency factor decides the order:
        // the fresh entry must outrank the year-old one, and both must
        // survive the relative score floor.
        let results = memory
            .search_memory("nebula backoff", 10, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "22222222-2222-4222-8222-222222222222");
        assert_eq!(results[1].id, "11111111-1111-4111-8111-111111111111");
        assert!(results[0].score > results[1].score);
        assert!(results[1].score > 0.0);
    }

    #[tokio::test]
    async fn test_ask_memory_content_truncation_options() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        // Content well over the 200-byte default preview.
        let long_content = format!("Comet pipeline detail: {}", "x".repeat(500));
        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Comet pipeline detail",
                &long_content,
            )
            .await
            .unwrap();

        // Default: truncated preview with ellipsis.
        let default_answer = memory.ask_memory("comet pipeline", None).await.unwrap();
        assert!(default_answer.contains("..."));
        assert!(!default_answer.contains(&"x".repeat(500)));

        // content_chars = 0: full content, no ellipsis.
        let full_answer = memory
            .ask_memory_with_options("comet pipeline", None, AskOptions { content_chars: 0 })
            .await
            .unwrap();
        assert!(full_answer.contains(&"x".repeat(500)));
        assert!(!full_answer.contains("..."));
    }

    #[tokio::test]
    async fn test_mark_used_boosts_ranking_and_persists() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        // Two entries with identical text so raw scores tie; only the
        // usage boost can separate them.
        let id_a = memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Orion cache policy",
                "The orion cache evicts least-recently-used pages first.",
            )
            .await
            .unwrap();
        let id_b = memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Orion cache policy",
                "The orion cache evicts least-recently-used pages first.",
            )
            .await
            .unwrap();

        // Mark entry B useful three times (also exercises prefix resolution).
        let prefix = &id_b[..8];
        for _ in 0..3 {
            let (resolved, count) = memory.mark_used(prefix).await.unwrap();
            assert_eq!(resolved, id_b);
            assert!(count >= 1);
        }

        // Unknown id is rejected.
        assert!(memory.mark_used("no-such-id").await.is_err());

        // Usage events persist to the sidecar and reload on reopen.
        drop(memory);
        let reopened = Mnemoria::open(&path).await.unwrap();
        let results = reopened
            .search_memory("orion cache eviction", 10, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id_b);
        assert_eq!(results[1].id, id_a);
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn test_entity_index_finds_paths_and_identifiers() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());

        let memory = Mnemoria::create(&path).await.unwrap();

        memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Config location",
                "The main config lives at E:/Anti-Gravity/opencode.json and uses the serde_json crate.",
            )
            .await
            .unwrap();

        memory
            .remember(
                "test-agent",
                EntryType::Warning,
                "Unrelated entry",
                "Nothing to see here.",
            )
            .await
            .unwrap();

        // Exact path lookup.
        let hits = memory
            .find_entities("E:/Anti-Gravity/opencode.json", 10)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].summary, "Config location");

        // Case-insensitive identifier lookup.
        let hits = memory.find_entities("SERDE_JSON", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        // Prefix lookup.
        let hits = memory.find_entities("opencode", 10).await.unwrap();
        assert_eq!(hits.len(), 1);

        // No match.
        let hits = memory.find_entities("nonexistent-thing", 10).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn test_get_by_id_or_prefix() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());
        let memory = Mnemoria::create(&path).await.unwrap();
        let id = memory
            .remember(
                "test-agent",
                EntryType::Discovery,
                "Prefix lookup",
                "Content",
            )
            .await
            .unwrap();

        assert_eq!(
            memory.get_by_id_or_prefix(&id).await.unwrap().unwrap().id,
            id
        );
        assert_eq!(
            memory
                .get_by_id_or_prefix(&id[..8])
                .await
                .unwrap()
                .unwrap()
                .id,
            id
        );
        assert!(
            memory
                .get_by_id_or_prefix("no-such")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_consolidate_supersedes_older_embedding_duplicates() {
        let temp_dir = create_temp_dir();
        let path = PathBuf::from(temp_dir.path());
        let import_path = path.join("consolidate-fixture.json");
        let export_path = path.join("consolidate-result.json");

        let older_id = "11111111-1111-4111-8111-111111111111";
        let newer_id = "22222222-2222-4222-8222-222222222222";
        let distinct_id = "33333333-3333-4333-8333-333333333333";
        let entries = serde_json::json!([
            {
                "id": older_id,
                "agent_name": "test-agent",
                "entry_type": "discovery",
                "summary": "Orion cache policy old",
                "content": "Orion cache uses strict LRU eviction policy",
                "embedding": [1.0, 0.0],
                "timestamp": 1000,
                "checksum": 0,
                "prev_checksum": 0
            },
            {
                "id": newer_id,
                "agent_name": "test-agent",
                "entry_type": "decision",
                "summary": "Orion cache policy current",
                "content": "Orion cache uses strict LRU eviction policy",
                "embedding": [1.0, 0.0],
                "timestamp": 2000,
                "checksum": 0,
                "prev_checksum": 0
            },
            {
                "id": distinct_id,
                "agent_name": "test-agent",
                "entry_type": "warning",
                "summary": "Credential storage",
                "content": "Credentials remain only in environment files",
                "embedding": [0.0, 1.0],
                "timestamp": 1500,
                "checksum": 0,
                "prev_checksum": 0
            }
        ]);
        std::fs::write(&import_path, serde_json::to_vec_pretty(&entries).unwrap()).unwrap();

        let memory = Mnemoria::create(&path).await.unwrap();
        assert_eq!(memory.import(&import_path).await.unwrap(), 3);

        let report = memory.consolidate(0.99, 2).await.unwrap();
        assert_eq!(report.entries_before, 3);
        assert_eq!(report.clusters_merged, 1);
        assert_eq!(report.superseded, 1);
        assert!(memory.verify().await.unwrap());

        memory.export(&export_path).await.unwrap();
        let exported: Vec<MemoryEntry> =
            serde_json::from_slice(&std::fs::read(&export_path).unwrap()).unwrap();
        assert_eq!(exported.len(), 4);
        assert!(exported.iter().any(|entry| {
            entry
                .summary
                .starts_with(&format!("SUPERSEDES {older_id} —"))
        }));
        assert!(!exported.iter().any(|entry| {
            entry
                .summary
                .starts_with(&format!("SUPERSEDES {newer_id} —"))
        }));

        let results = memory
            .search_memory("Orion cache eviction policy", 10, None)
            .await
            .unwrap();
        assert!(results.iter().any(|result| result.id == newer_id));
        assert!(!results.iter().any(|result| result.id == older_id));
        assert!(memory.get(distinct_id).await.unwrap().is_some());
    }
}
