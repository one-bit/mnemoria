# Mnemoria

[![CI](https://github.com/adevwithpurpose/mnemoria/actions/workflows/ci.yml/badge.svg)](https://github.com/adevwithpurpose/mnemoria/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnemoria.svg)](https://crates.io/crates/mnemoria)
[![docs.rs](https://docs.rs/mnemoria/badge.svg)](https://docs.rs/mnemoria)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-181717?logo=githubsponsors)](https://github.com/sponsors/one-bit)

Mnemoria is a **memory storage system for AI agents**. It provides persistent, searchable memory that AI assistants can use to remember information across conversations and sessions. Perfect for Claude, GPT, Cursor, or any AI tool that needs long-term context.

## Library Usage

```rust
use mnemoria::{Mnemoria, EntryType};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), mnemoria::Error> {
    // Create a new memory store
    let memory = Mnemoria::create(Path::new("./my-memories")).await?;

    // Store a memory
    let id = memory.remember(
        "my-agent",
        EntryType::Discovery,
        "Rust async patterns",
        "Use tokio::spawn for CPU-bound work inside async contexts",
    ).await?;

    // Search by meaning (hybrid BM25 + semantic)
    let results = memory.search_memory("async concurrency", 5, None).await?;
    for result in &results {
        println!("[{}] {} (score: {:.3})", result.entry.entry_type, result.entry.summary, result.score);
    }

    // Retrieve by ID
    let entry = memory.get(&id).await?;
    Ok(())
}
```

## Support this project

If this project has been helpful to you, you are welcome to sponsor it.
Sponsorship helps me spend more time maintaining it, fixing bugs, and
building new features.

No pressure at all - starring the repo, sharing it, or giving feedback also
means a lot.

[Become a sponsor](https://github.com/sponsors/one-bit)

## Features

- **Semantic Search** - Find memories by meaning, not just keywords
- **Full-Text Search** - BM25-powered keyword search  
- **Hybrid Search** - Combines both approaches via Reciprocal Rank Fusion
- **Git-Friendly** - Append-only binary format, version control safe
- **Corruption Protection** - CRC32 checksum chain with crash recovery
- **Supersede-Aware Recall** - Tombstoned memories and tombstones stay out of search and ask results
- **Adaptive Ranking** - Type importance, recency decay, bounded usage boosts, and a relative score floor
- **Full-Fidelity Output** - JSON search output includes full IDs and content; ask/search previews are configurable
- **Entity Index** - Mechanical lookup of paths, UUIDs, filenames, and compound identifiers without an LLM
- **Semantic Consolidation** - Cluster stored embeddings, keep the newest duplicate, and append audit-safe tombstones
- **Superseded Pruning** - Optional physical cleanup with checksum-chain relinking
- **Unlimited Size** - Only bounded by disk space

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- ~130MB for embedding model (downloaded on first use)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/adevwithpurpose/mnemoria
cd mnemoria

# Build and install
cargo install --path .
```

### Via crates.io

```bash
cargo install mnemoria
```

## Quick Start

```bash
# 1. Initialize a new memory store in the current directory
mnemoria init

# Or specify a path
mnemoria --path /path/to/project init

# 2. Add a memory entry
mnemoria add --type discovery \
  --summary "Found optimal async pattern for file I/O" \
  "Use tokio's fs::File with spawn_blocking for CPU-intensive work..."

# 3. Search your memories  
mnemoria search "async file operations"

# 4. Ask questions about your memories
mnemoria ask "what async patterns have I discovered?"
```

## Commands

| Command        | Description                                       |
| -------------- | ------------------------------------------------- |
| `init`           | Create a new memory store                         |
| `add`            | Add a memory entry                                |
| `search`         | Search memories by keyword or semantic similarity |
| `ask`            | Ask a natural language question                   |
| `stats`          | Show memory statistics                            |
| `verify`         | Verify integrity (detect corruption)              |
| `timeline`       | View memories chronologically                     |
| `rebuild-index`  | Rebuild the search index                          |
| `get`            | Retrieve a full entry by UUID or unambiguous prefix |
| `compact`        | Rebuild the store; optionally prune superseded entries |
| `consolidate`    | Supersede older near-duplicates using stored embeddings |
| `entities`       | Find entries mentioning a path, UUID, or identifier |
| `mark-used`      | Record useful recall in the append-only usage sidecar |
| `export`         | Export memories to JSON                           |
| `import`         | Import memories from JSON                         |

## Entry Types

When adding memories, you can categorize them:

- `intent` - Goals and intentions
- `discovery` - Things you learned
- `decision` - Decisions made
- `problem` - Problems encountered
- `solution` - Solutions found
- `pattern` - Recurring patterns
- `warning` - Warnings to remember
- `success` - Successes/outcomes
- `refactor` - Refactoring notes
- `bugfix` - Bug fixes applied
- `feature` - Features implemented

## Git Usage

Mnemoria uses an append-only binary format designed for version control. You can commit your `mnemoria/` directory directly to track memory history alongside your code:

```bash
# Track memories in git (recommended for most projects)
git add mnemoria/
git commit -m "add project memories"
```

For large memory stores, use Git LFS:

```bash
git lfs track "mnemoria/log.bin"
git add .gitattributes mnemoria/
```

If you prefer not to track memories in version control:

```bash
echo "mnemoria/" >> .gitignore
```

## Storage Format

```
mnemoria/
├── log.bin           # Append-only binary log
├── manifest.json     # Metadata and checksums
└── mnemoria.lock     # Advisory file lock
```

The search index is rebuilt on each open and is not stored in git.

## Architecture

- **Storage**: rkyv binary serialization (zero-copy)
- **Full-Text**: Tantivy (BM25)
- **Embeddings**: model2vec (1024-dim static retrieval by default, CPU-only)
- **Similarity**: simsimd (SIMD-accelerated)


## Performance

Benchmarks run with [Criterion.rs](https://github.com/bheisler/criterion.rs)
(`cargo bench --bench api_perf`). Results below are median values.

### Test Environment

| Component | Details |
|-----------|---------|
| CPU | AMD Ryzen 9 9950X3D 16-Core (32 threads), up to 5.76 GHz, 128 MB L3 cache |
| RAM | 94 GB DDR5 |
| Storage | NVMe SSD (Samsung 960 EVO 1TB / Crucial T705 4TB) |
| OS | Fedora 43 (Linux 6.18.8, x86_64) |
| Rust | 1.93.1 (stable) |

### Search Latency (hybrid: BM25 + semantic via RRF)

| Entries | Latency |
|---------|---------|
| 1,000 | ~95 us |
| 5,000 | ~341 us |
| 10,000 | ~756 us |

### Write Throughput (200-entry batches)

| Durability Mode | Throughput |
|-----------------|------------|
| `Fsync` (default) | ~9,900 entries/sec |
| `FlushOnly` | ~9,990 entries/sec |
| `None` | ~9,760 entries/sec |

### Get by ID

| Entries | Cached (in-memory) | Disk Scan (baseline) |
|---------|--------------------|----------------------|
| 1,000 | ~2.5 us | ~174 us |
| 5,000 | ~2.4 us | ~982 us |

### Timeline

| Entries | Cached (in-memory) | Disk Scan (baseline) |
|---------|--------------------|----------------------|
| 1,000 | ~14.5 us | ~177 us |
| 5,000 | ~14.4 us | ~994 us |

### Model comparison (v0.4.1)

An eight-model sweep on Windows 11 used 30 documents in 10 confusable clusters and 30 queries. The benchmark computes Hugging Face cache SHA-256 hashes in-process, so it runs on Windows without the Unix sha256sum utility. It auto-converts sentence-transformers static-embedding models to the flat Model2Vec layout.

| Model | Semantic P@1 | Semantic MRR | Hybrid P@1 | Hybrid MRR | Embed | Search | Write |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **static-retrieval-mrl-en-v1** | **96.7%** | **0.9833** | 76.7% | 0.8233 | 83.4 us | 1.81 ms | 57.1/s |
| potion-multilingual-128M | 93.3% | 0.9583 | **76.7%** | **0.8250** | 77.8 us | 1.84 ms | 46.9/s |
| potion-base-32M | 93.3% | 0.9667 | 76.7% | 0.8233 | 42.6 us | 1.93 ms | 52.7/s |
| potion-base-8M | 90.0% | 0.9444 | 73.3% | 0.8067 | 45.0 us | 1.70 ms | 56.4/s |
| potion-retrieval-32M | 96.7% | 0.9778 | 73.3% | 0.7956 | 73.5 us | 1.73 ms | 54.0/s |
| potion-code-16M-v2 | 90.0% | 0.9500 | 70.0% | 0.7800 | 59.8 us | 1.45 ms | 63.7/s |
| potion-base-4M | 90.0% | 0.9354 | 70.0% | 0.7778 | 61.4 us | 1.32 ms | 54.8/s |
| potion-base-2M | 86.7% | 0.9044 | 70.0% | 0.7800 | 50.0 us | 1.59 ms | 64.0/s |

The default is now **sentence-transformers/static-retrieval-mrl-en-v1**: it wins pure semantic recall (MRR 0.9833, P@1 96.7%), ties hybrid recall with potion-base-32M (0.8233), and — because it is a 1024-dim static embedding — loads roughly 2x faster per CLI invocation than the old default. potion-multilingual-128M matches hybrid MRR (0.8250) but is 4x the size with no semantic gain. Run both benchmark suites with:

```bash
cargo bench --bench api_perf
cargo bench --bench model_comparison
```

## License

MIT License. See `LICENSE` for details.

## Repository

https://github.com/adevwithpurpose/mnemoria
