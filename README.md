# Mnemoria

[![CI](https://github.com/one-bit/mnemoria/actions/workflows/ci.yml/badge.svg)](https://github.com/one-bit/mnemoria/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/mnemoria.svg)](https://crates.io/crates/mnemoria)

Mnemoria is a **memory storage system for AI agents**. It provides persistent, searchable memory that AI assistants can use to remember information across conversations and sessions. Perfect for Claude, GPT, Cursor, or any AI tool that needs long-term context.

## Features

- **Semantic Search** - Find memories by meaning, not just keywords
- **Full-Text Search** - BM25-powered keyword search  
- **Hybrid Search** - Combines both approaches via Reciprocal Rank Fusion
- **Git-Friendly** - Append-only binary format, version control safe
- **Corruption Protection** - CRC32 checksum chain with crash recovery
- **Unlimited Size** - Only bounded by disk space

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- ~30MB for embedding model (downloaded on first use)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/one-bit/mnemoria
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
| `compact`        | Remove corrupt entries and rewrite log             |
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
- **Embeddings**: model2vec (256-dim, CPU-only)
- **Similarity**: simsimd (SIMD-accelerated)

See `ARCHITECTURE.md` and `PRD.md` for detailed design.

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

To run benchmarks yourself:

```bash
cargo bench --bench api_perf
```

## License

MIT License. See `LICENSE` for details.

## Repository

https://github.com/one-bit/mnemoria
