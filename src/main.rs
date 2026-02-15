use clap::{Parser, Subcommand};
use mnemoria::{APP_NAME, Config, DEFAULT_MODEL_ID, EntryType, Mnemoria};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "mnemoria")]
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
    Search {
        query: String,

        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Filter results to only this agent's entries.
        #[arg(short = 'a', long = "agent")]
        agent_name: Option<String>,
    },
    Ask {
        question: String,

        /// Filter results to only this agent's entries.
        #[arg(short = 'a', long = "agent")]
        agent_name: Option<String>,
    },
    Stats {},
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
    },
    RebuildIndex {},
    Compact {},
    Export {
        output: String,
    },
    Import {
        input: String,
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
        Commands::Search {
            query,
            limit,
            agent_name,
        } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let results = memory
                .search_memory(&query, limit, agent_name.as_deref())
                .await?;
            println!("Found {} results:", results.len());
            for (i, result) in results.iter().enumerate() {
                println!(
                    "{}. [{}] ({}) {} (score: {:.3})",
                    i + 1,
                    result.entry.entry_type,
                    result.entry.agent_name,
                    result.entry.summary,
                    result.score
                );
            }
        }
        Commands::Ask {
            question,
            agent_name,
        } => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let answer = memory.ask_memory(&question, agent_name.as_deref()).await?;
            println!("{answer}");
        }
        Commands::Stats {} => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            let stats = memory.memory_stats().await?;
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
            println!("Timeline ({} entries):", entries.len());
            for (i, entry) in entries.iter().enumerate() {
                println!(
                    "{}. [{}] ({}) {} - {}",
                    i + 1,
                    entry.entry_type,
                    entry.agent_name,
                    entry.summary,
                    entry.timestamp
                );
            }
        }
        Commands::RebuildIndex {} => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            memory.rebuild_index().await?;
            println!("Index rebuilt successfully");
        }
        Commands::Compact {} => {
            let memory = Mnemoria::open_with_config(&memory_path, config).await?;
            memory.compact().await?;
            println!("Memory compacted successfully");
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
    }

    Ok(())
}
