use clap::{Parser, Subcommand};
use mnemoria::{APP_NAME, Config, DEFAULT_MODEL_ID, EntryType, Mnemoria};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Short (8-char) prefix of an entry UUID for human-readable output.
///
/// Full IDs are available via the `--json` flag on supported commands.
fn short_id(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}

/// Truncate content to at most `max_bytes` bytes (cut on a UTF-8 char
/// boundary) for preview output, matching the library's ask-memory
/// truncation semantics.
///
/// `max_bytes == 0` returns the full content. A trailing `...` marks a
/// truncated preview.
fn preview_content(content: &str, max_bytes: usize) -> String {
    if max_bytes == 0 || content.len() <= max_bytes {
        return content.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &content[..end])
}

/// JSON-friendly view of a memory entry that omits the embedding vector.
#[derive(Serialize)]
struct EntryView<'a> {
    id: &'a str,
    agent_name: &'a str,
    entry_type: EntryType,
    summary: &'a str,
    content: &'a str,
    timestamp: i64,
    checksum: u32,
    prev_checksum: u32,
}

impl<'a> From<&'a mnemoria::MemoryEntry> for EntryView<'a> {
    fn from(e: &'a mnemoria::MemoryEntry) -> Self {
        Self {
            id: &e.id,
            agent_name: &e.agent_name,
            entry_type: e.entry_type,
            summary: &e.summary,
            content: &e.content,
            timestamp: e.timestamp,
            checksum: e.checksum,
            prev_checksum: e.prev_checksum,
        }
    }
}

/// JSON-friendly view of a search result.
#[derive(Serialize)]
struct SearchResultView<'a> {
    id: &'a str,
    score: f32,
    entry: EntryView<'a>,
}

#[derive(Parser)]
#[command(name = "mnemoria")]
#[command(version)]
#[command(about = "A git-friendly memory storage CLI for AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, default_value = ".")]
    path: PathBuf,

    /// HuggingFace model ID for the model2vec embedding model.
    #[arg(short, long, default_value = DEFAULT_MODEL_ID)]
    model: String,
}

#[derive(Subcommand)]
enum Commands {
    Init {
        name: Option<String>,
    },
    Add {
        /// Name of the agent storing the memory.
        #[arg(short = 'a', long = "agent")]
        agent_name: String,

        #[arg(short = 't', long = "type", value_enum, default_value_t = EntryType::Discovery)]
        entry_type: EntryType,

        #[arg(short, long)]
        summary: String,

        content: String,
    },
    Get {
        /// Entry id (full UUID or an unambiguous prefix).
        id: String,

        /// Output full entry data as JSON (embedding omitted).
        #[arg(long)]
        json: bool,
    },
    Search {
        query: String,

        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Filter results to only this agent's entries.
        #[arg(short = 'a', long = "agent")]
        agent_name: Option<String>,

        /// Output results as JSON (full entry data, embeddings omitted).
        #[arg(long)]
        json: bool,

        /// Include a content preview under each human-readable result.
        /// Ignored with --json (JSON always carries the full content).
        #[arg(long)]
        with_content: bool,

        /// Content preview length in bytes for --with-content (0 = full).
        #[arg(long, default_value = "200")]
        content_chars: usize,
    },
    Ask {
        question: String,

        /// Filter results to only this agent's entries.
        #[arg(short = 'a', long = "agent")]
        agent_name: Option<String>,

        /// Bytes of each entry's content to include (0 = full content).
        #[arg(long, default_value = "200")]
        content_chars: usize,
    },
    Stats {
        /// Output statistics as JSON.
        #[arg(long)]
        json: bool,
    },
    Verify {},
    Timeline {
        #[arg(short, long, default_value = "20")]
        limit: usize,

        #[arg(short, long)]
        since: Option<i64>,

        #[arg(short, long)]
        until: Option<i64>,

        #[arg(short, long, default_value_t = true)]
        reverse: bool,

        /// Filter entries to only this agent.
        #[arg(short = 'a', long = "agent")]
        agent_name: Option<String>,

        /// Output entries as JSON (full entry data, embeddings omitted).
        #[arg(long)]
        json: bool,
    },
    RebuildIndex {},
    Compact {
        /// Physically remove entries hidden by SUPERSEDES tombstones (and the
        /// tombstones themselves). The checksum chain is relinked.
        #[arg(long)]
        prune_superseded: bool,
    },
    /// Merge near-duplicate entries detected via stored-embedding cosine
    /// similarity, so recall has one best copy per idea. For each cluster of
    /// at least --min-size members, the newest entry is kept and the older
    /// copies are superseded with SUPERSEDES tombstones. The append-only log
    /// is preserved; run compact --prune-superseded to remove them.
    Consolidate {
        /// Minimum cosine similarity (0.0..=1.0) for two entries to count as
        /// near-duplicates. Higher is stricter. Default 0.90.
        #[arg(long, default_value = "0.90")]
        threshold: f32,

        /// Minimum cluster size before it is consolidated. Use 2 to merge
        /// duplicate pairs (default).
        #[arg(long, default_value = "2")]
        min_size: usize,
    },
    Export {
        output: String,
    },
    Import {
        input: String,
    },
    /// Record that an entry was useful, boosting its future search ranking.
    ///
    /// Usage events are appended to an append-only usage.jsonl sidecar; the
    /// entry log itself is not modified. Each use adds a bounded score boost
    /// (saturating after 5 uses).
    MarkUsed {
        /// Id of the entry to mark as useful (full UUID or 8-char prefix).
        id: String,
    },
    /// Find entries that mention a given entity (path, UUID, identifier).
    ///
    /// Uses the mechanical entity index built from entry summaries and
    /// content. Matching is case-insensitive; a term that is not an exact
    /// token match falls back to substring containment.
    Entities {
        /// Entity term to look up (e.g. a file path, UUID, or crate name).
        term: String,

        /// Maximum number of entries to return.
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Output entries as JSON (full entry data, embeddings omitted).
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let memory_path = if cli.path.is_dir() {
        cli.path.join(APP_NAME)
    } else {
        cli.path.clone()
    };

    let config = Config {
        model_id: cli.model,
        ..Config::default()
    };

    match cli.command {
        Commands::Init { name } => {
            let dir_name = name.as_deref().unwrap_or(APP_NAME);
            let init_path = cli.path.join(dir_name);
            let memory = Mnemoria::create_with_config(&init_path, config).await?;
            println!("Created memory at {init_path:?}");
            let stats = memory.memory_stats().await?;
            println!("Total entries: {}", stats.total_entries);
        }
        Commands::Add {
            agent_name,
            entry_type,
            summary,
            content,
        } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let id = memory
                .remember(&agent_name, entry_type, &summary, &content)
                .await?;
            println!("Added entry: {id}");
        }
        Commands::Get { id, json } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let entry = memory.get_by_id_or_prefix(&id).await?.ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Entry not found: {id}"),
                )
            })?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&EntryView::from(&entry))?
                );
            } else {
                println!("ID: {}", entry.id);
                println!("Type: {}", entry.entry_type);
                println!("Agent: {}", entry.agent_name);
                println!("Timestamp: {}", entry.timestamp);
                println!("Summary: {}", entry.summary);
                println!(
                    "Content:
{}",
                    entry.content
                );
            }
        }
        Commands::Search {
            query,
            limit,
            agent_name,
            json,
            with_content,
            content_chars,
        } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let results = memory
                .search_memory(&query, limit, agent_name.as_deref())
                .await?;
            if json {
                let views: Vec<SearchResultView> = results
                    .iter()
                    .map(|r| SearchResultView {
                        id: &r.id,
                        score: r.score,
                        entry: EntryView::from(&r.entry),
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&views)?);
            } else {
                println!("Found {} results:", results.len());
                for (i, result) in results.iter().enumerate() {
                    println!(
                        "{}. [{}] ({}) {} {} (score: {:.3})",
                        i + 1,
                        result.entry.entry_type,
                        result.entry.agent_name,
                        short_id(&result.id),
                        result.entry.summary,
                        result.score
                    );
                    if with_content {
                        let preview = preview_content(&result.entry.content, content_chars);
                        println!("   {preview}");
                    }
                }
            }
        }
        Commands::Ask {
            question,
            agent_name,
            content_chars,
        } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let answer = memory
                .ask_memory_with_options(
                    &question,
                    agent_name.as_deref(),
                    mnemoria::AskOptions { content_chars },
                )
                .await?;
            println!("{answer}");
        }
        Commands::Stats { json } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let stats = memory.memory_stats().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("Memory Statistics:");
                println!("  Total entries: {}", stats.total_entries);
                println!("  File size: {} bytes", stats.file_size_bytes);
                if let Some(oldest) = stats.oldest_timestamp {
                    println!("  Oldest entry: {oldest}");
                }
                if let Some(newest) = stats.newest_timestamp {
                    println!("  Newest entry: {newest}");
                }
            }
        }
        Commands::Verify {} => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let valid = memory.verify().await?;
            if valid {
                println!("Memory verification passed - checksums are valid");
            } else {
                eprintln!("Memory verification FAILED - corruption detected!");
                std::process::exit(1);
            }
        }
        Commands::Timeline {
            limit,
            since,
            until,
            reverse,
            agent_name,
            json,
        } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let entries = memory
                .timeline(mnemoria::TimelineOptions {
                    limit,
                    since,
                    until,
                    reverse,
                    agent_name,
                })
                .await?;
            if json {
                let views: Vec<EntryView> = entries.iter().map(EntryView::from).collect();
                println!("{}", serde_json::to_string_pretty(&views)?);
            } else {
                println!("Timeline ({} entries):", entries.len());
                for (i, entry) in entries.iter().enumerate() {
                    println!(
                        "{}. [{}] ({}) {} {} - {}",
                        i + 1,
                        entry.entry_type,
                        entry.agent_name,
                        short_id(&entry.id),
                        entry.summary,
                        entry.timestamp
                    );
                }
            }
        }
        Commands::RebuildIndex {} => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            memory.rebuild_index().await?;
            println!("Index rebuilt successfully");
        }
        Commands::Compact { prune_superseded } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let report = memory
                .compact_with_options(mnemoria::CompactOptions { prune_superseded })
                .await?;
            println!(
                "Memory compacted successfully ({} -> {} entries, {} pruned)",
                report.entries_before, report.entries_after, report.pruned_superseded
            );
        }
        Commands::Consolidate {
            threshold,
            min_size,
        } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let report = memory.consolidate(threshold, min_size).await?;
            println!(
                "Consolidated {} clusters ({} entries before, {} superseded)",
                report.clusters_merged, report.entries_before, report.superseded
            );
        }
        Commands::Export { output } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            memory.export(Path::new(&output)).await?;
            println!("Exported to {output}");
        }
        Commands::Import { input } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let count = memory.import(Path::new(&input)).await?;
            println!("Imported {count} entries from {input}");
        }
        Commands::MarkUsed { id } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let (resolved_id, count) = memory.mark_used(&id).await?;
            println!(
                "Marked entry {} as used (total uses: {count})",
                short_id(&resolved_id)
            );
        }
        Commands::Entities { term, limit, json } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let entries = memory.find_entities(&term, limit).await?;
            if json {
                let views: Vec<EntryView> = entries.iter().map(EntryView::from).collect();
                println!("{}", serde_json::to_string_pretty(&views)?);
            } else {
                println!(
                    "Entries mentioning entity \"{term}\" ({} found):",
                    entries.len()
                );
                for (i, entry) in entries.iter().enumerate() {
                    println!(
                        "{}. [{}] ({}) {} {} - {}",
                        i + 1,
                        entry.entry_type,
                        entry.agent_name,
                        short_id(&entry.id),
                        entry.summary,
                        entry.timestamp
                    );
                }
            }
        }
    }

    Ok(())
}
