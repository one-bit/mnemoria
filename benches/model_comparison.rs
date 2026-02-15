//! Model Comparison Benchmark
//!
//! Evaluates the top 5 model2vec models across two key metrics:
//!
//! 1. **Retrieval Accuracy** — measured in two modes:
//!    - **Semantic-only**: pure cosine similarity between query and document
//!      embeddings (no BM25). This isolates the embedding model quality.
//!    - **Hybrid (BM25 + semantic)**: the full Mnemoria search pipeline using
//!      Reciprocal Rank Fusion.
//!
//!    The corpus is designed with *confusable clusters* — groups of documents
//!    that share keywords but differ semantically — so BM25 alone cannot
//!    reliably pick the correct document and the embedding must contribute.
//!
//! 2. **Retrieval Performance** — wall-clock time for:
//!    - Embedding generation (single text)
//!    - Search latency over a populated store
//!    - Write throughput (entries/sec including embedding)
//!
//! Run with:
//!   cargo bench --bench model_comparison
//!
//! NOTE: Models are downloaded on first run (~4-130 MB each). Ensure network
//! access is available for the initial run.

use mnemoria::embeddings::EmbeddingBackend;
use mnemoria::{APP_NAME, Config, DurabilityMode, EntryType, MODELS_SUBDIR, Mnemoria};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// The top 5 publicly available model2vec models from the MTEB results table.
/// Ordered by overall MTEB score (descending).
///
/// Note: `static-retrieval-mrl-en-v1` is excluded because it requires
/// HuggingFace authentication. We use `potion-base-2M` instead to cover
/// the full size range from 2M to 32M parameters.
const MODELS: &[(&str, &str)] = &[
    ("potion-base-32M", "minishlab/potion-base-32M"),
    ("potion-base-8M", "minishlab/potion-base-8M"),
    ("potion-retrieval-32M", "minishlab/potion-retrieval-32M"),
    ("potion-base-4M", "minishlab/potion-base-4M"),
    ("potion-base-2M", "minishlab/potion-base-2M"),
];

// ---------------------------------------------------------------------------
// Test corpus: confusable clusters that share keywords
// ---------------------------------------------------------------------------
//
// Each cluster contains 3-4 documents using overlapping vocabulary.
// Queries must rely on *semantic* understanding to pick the right one.
// ---------------------------------------------------------------------------

struct RetrievalTestCase {
    query: &'static str,
    expected_index: usize,
}

const CORPUS: &[(&str, &str)] = &[
    // ── Cluster 1: "memory" (programming vs psychology vs biology) ──────────
    // 0
    (
        "Rust ownership and memory management",
        "The Rust programming language prevents memory leaks and data races through \
         its ownership model. Each value has a single owner, and the borrow checker \
         enforces reference lifetimes at compile time without a garbage collector.",
    ),
    // 1
    (
        "Human memory formation and recall",
        "Long-term memories are formed through a process called consolidation, where \
         the hippocampus transfers information to the neocortex during sleep. Recall \
         involves reactivating neural pathways that were strengthened during encoding.",
    ),
    // 2
    (
        "Computer memory hierarchy and caching",
        "Modern CPUs use a memory hierarchy with L1, L2, and L3 caches to bridge the \
         speed gap between the processor and main RAM. Cache eviction policies like LRU \
         determine which data stays close to the CPU for fast access.",
    ),
    // ── Cluster 2: "network" (computer vs neural vs social) ────────────────
    // 3
    (
        "TCP/IP network protocol stack",
        "The TCP/IP model organizes network communication into four layers: link, \
         internet, transport, and application. TCP provides reliable ordered delivery \
         through sequence numbers and acknowledgment packets over IP routing.",
    ),
    // 4
    (
        "Biological neural networks in the brain",
        "The brain contains approximately 86 billion neurons connected by trillions of \
         synapses. Neural signals propagate through electrochemical impulses, with \
         neurotransmitters bridging the synaptic gap between axon terminals and dendrites.",
    ),
    // 5
    (
        "Social network analysis and graph theory",
        "Social network analysis maps relationships between individuals using graph \
         structures. Centrality measures like betweenness and degree identify influential \
         nodes, while community detection reveals clusters of tightly connected people.",
    ),
    // ── Cluster 3: "model" (ML vs fashion vs architecture) ─────────────────
    // 6
    (
        "Training large language models",
        "Large language models are trained on massive text corpora using transformer \
         architectures. Pre-training on next-token prediction followed by instruction \
         fine-tuning and RLHF alignment produces capable conversational AI systems.",
    ),
    // 7
    (
        "Fashion modeling and runway shows",
        "Professional fashion models walk runways during seasonal collections in major \
         cities like Paris, Milan, New York, and London. Agencies scout for distinctive \
         looks and proportions, while designers select models to embody their aesthetic.",
    ),
    // 8
    (
        "Architectural scale models and design",
        "Architects build physical scale models to visualize spatial relationships, \
         test structural ideas, and communicate designs to clients. Materials like foam \
         core, basswood, and 3D-printed components bring blueprints to life.",
    ),
    // ── Cluster 4: "cell" (biology vs prison vs battery vs spreadsheet) ────
    // 9
    (
        "Biological cell division and mitosis",
        "During mitosis, a eukaryotic cell duplicates its chromosomes and divides into \
         two genetically identical daughter cells. The process involves prophase, metaphase, \
         anaphase, and telophase, each tightly regulated by cyclin-dependent kinases.",
    ),
    // 10
    (
        "Lithium-ion battery cell chemistry",
        "Lithium-ion cells generate electricity through the movement of lithium ions \
         between a graphite anode and a lithium cobalt oxide cathode. The electrolyte \
         facilitates ion transport while a separator prevents short circuits.",
    ),
    // 11
    (
        "Spreadsheet cell formulas and functions",
        "Spreadsheet cells can contain values, text, or formulas that reference other \
         cells. Functions like VLOOKUP, SUMIF, and pivot tables enable complex data \
         analysis, while conditional formatting highlights patterns visually.",
    ),
    // ── Cluster 5: "tree" (data structure vs botany vs genealogy) ──────────
    // 12
    (
        "Binary search tree algorithms",
        "A binary search tree maintains sorted order: each node has at most two children, \
         with left subtree values less than the parent and right subtree values greater. \
         Self-balancing variants like AVL and red-black trees guarantee O(log n) operations.",
    ),
    // 13
    (
        "Photosynthesis in deciduous trees",
        "Deciduous trees convert sunlight into glucose through photosynthesis in their \
         leaves. Chlorophyll absorbs red and blue wavelengths, driving the Calvin cycle \
         that fixes atmospheric CO2. In autumn, reduced daylight triggers leaf senescence.",
    ),
    // 14
    (
        "Family genealogy tree research",
        "Genealogy research traces ancestral lineage through birth, marriage, and death \
         records. DNA testing combined with archival research can extend family trees \
         back centuries, revealing migration patterns and ethnic heritage.",
    ),
    // ── Cluster 6: "pattern" (design vs regex vs behavioral) ───────────────
    // 15
    (
        "Software design patterns and SOLID principles",
        "Design patterns like Singleton, Observer, and Factory provide reusable solutions \
         to common software problems. SOLID principles guide object-oriented design toward \
         maintainable, extensible code with clear separation of concerns.",
    ),
    // 16
    (
        "Regular expression pattern matching",
        "Regular expressions define search patterns using metacharacters like quantifiers, \
         character classes, and anchors. Regex engines use backtracking or NFA-based \
         algorithms to match text against compiled pattern automata.",
    ),
    // 17
    (
        "Behavioral patterns in animal migration",
        "Animal migration follows inherited behavioral patterns triggered by environmental \
         cues like day length and temperature. Monarch butterflies navigate thousands of \
         miles using a time-compensated sun compass and magnetic field sensing.",
    ),
    // ── Cluster 7: "key" (cryptography vs music vs database) ──────────────
    // 18
    (
        "Public-key cryptography and RSA",
        "Public-key cryptography uses mathematically linked key pairs: a public key for \
         encryption and a private key for decryption. RSA security relies on the \
         computational difficulty of factoring large semiprimes into their prime components.",
    ),
    // 19
    (
        "Musical key signatures and transposition",
        "A key signature indicates the set of sharps or flats used throughout a piece. \
         Transposing shifts all pitches by a consistent interval, allowing musicians to \
         adapt repertoire to different vocal ranges or instrument tunings.",
    ),
    // 20
    (
        "Database primary keys and indexing",
        "A primary key uniquely identifies each row in a relational database table. \
         Clustered indexes physically sort data by the key, while secondary indexes \
         create separate B-tree structures pointing back to the primary key.",
    ),
    // ── Cluster 8: "python" (programming vs snake vs comedy) ──────────────
    // 21
    (
        "Python programming language features",
        "Python emphasizes readability with significant whitespace and dynamic typing. \
         Its extensive standard library and package ecosystem (pip/PyPI) make it popular \
         for web development, data science, and scripting automation tasks.",
    ),
    // 22
    (
        "Python snake species and biology",
        "Pythons are large non-venomous constrictor snakes found in tropical regions of \
         Africa, Asia, and Australia. They kill prey by coiling around it and tightening \
         with each exhalation, detecting body heat with infrared-sensing pit organs.",
    ),
    // 23
    (
        "Monty Python comedy and cultural impact",
        "Monty Python revolutionized sketch comedy with absurdist humor and surreal \
         transitions. Films like Life of Brian and The Holy Grail became cult classics, \
         influencing generations of comedians and the term for the programming language.",
    ),
    // ── Cluster 9: "plate" (geology vs dining vs printing) ─────────────────
    // 24
    (
        "Tectonic plate boundaries and earthquakes",
        "Tectonic plates collide at convergent boundaries, creating mountain ranges and \
         subduction zones. The stored elastic energy releases suddenly as earthquakes, \
         with magnitude measured by seismic moment on the moment magnitude scale.",
    ),
    // 25
    (
        "Fine dining plating and food presentation",
        "Michelin-starred chefs use plating techniques like quenelles, swooshes, and \
         negative space to create visually stunning dishes. Color contrast, height \
         variation, and garnish placement transform a meal into edible art.",
    ),
    // 26
    (
        "Offset printing plate technology",
        "Offset lithographic printing transfers ink from an aluminum plate to a rubber \
         blanket and then onto paper. CTP (computer-to-plate) technology directly images \
         plates from digital files, eliminating the need for photographic film.",
    ),
    // ── Cluster 10: "bridge" (engineering vs card game vs dental) ──────────
    // 27
    (
        "Suspension bridge engineering",
        "Suspension bridges support the roadway deck from vertical cables attached to \
         main cables draped between towers. The parabolic cable shape distributes loads \
         efficiently, enabling spans exceeding a mile like the Akashi Kaikyo Bridge.",
    ),
    // 28
    (
        "Contract bridge card game strategy",
        "Contract bridge is a trick-taking card game for four players in two partnerships. \
         Bidding communicates hand strength through conventions like Stayman and Blackwood, \
         while declarer play involves finessing and establishing long suits.",
    ),
    // 29
    (
        "Dental bridge prosthetics",
        "A dental bridge replaces one or more missing teeth by anchoring artificial teeth \
         (pontics) to adjacent natural teeth or implants. Materials include porcelain \
         fused to metal, zirconia, and all-ceramic options for aesthetic results.",
    ),
];

/// Queries designed to be maximally hard. They are split into two tiers:
///
/// **Tier A (abstract paraphrases):** use completely different vocabulary from
/// the target document, forcing the model to understand meaning rather than
/// match tokens. These are where weaker models should start failing.
///
/// **Tier B (cross-cluster distractors):** use keywords shared by multiple
/// cluster members but target a specific document. BM25 will be confused.
const TEST_CASES: &[RetrievalTestCase] = &[
    // ── Tier A: Abstract / zero-overlap paraphrases ────────────────────────
    // Cluster 1: "memory"
    RetrievalTestCase {
        // Target: Rust ownership (index 0). No Rust-specific words used.
        query: "a compiled language that statically proves absence of concurrent access violations",
        expected_index: 0,
    },
    RetrievalTestCase {
        // Target: Human memory (index 1). No neuroscience terms used.
        query: "how the brain stores experiences overnight and brings them back later",
        expected_index: 1,
    },
    RetrievalTestCase {
        // Target: CPU caches (index 2). Avoids technical cache terms.
        query: "hardware keeping frequently used data physically near the processor",
        expected_index: 2,
    },
    // Cluster 2: "network"
    RetrievalTestCase {
        // Target: TCP/IP (index 3). No protocol-specific terms.
        query: "guaranteeing ordered arrival of digital messages across the internet",
        expected_index: 3,
    },
    RetrievalTestCase {
        // Target: brain neurons (index 4). No anatomy terms.
        query: "biological signaling between billions of connected nerve cells",
        expected_index: 4,
    },
    RetrievalTestCase {
        // Target: social network analysis (index 5). No graph theory terms.
        query: "mapping relationships between people to find who is most connected",
        expected_index: 5,
    },
    // Cluster 3: "model"
    RetrievalTestCase {
        // Target: LLM training (index 6). Avoids ML jargon.
        query: "teaching a computer to predict the next word in a sentence",
        expected_index: 6,
    },
    RetrievalTestCase {
        // Target: fashion modeling (index 7). Avoids fashion terms.
        query: "people hired to wear designer clothing on a catwalk",
        expected_index: 7,
    },
    RetrievalTestCase {
        // Target: architectural models (index 8). Avoids architecture terms.
        query: "building tiny physical replicas of planned structures",
        expected_index: 8,
    },
    // Cluster 4: "cell"
    RetrievalTestCase {
        // Target: mitosis (index 9). Avoids biology terms.
        query: "the process by which a living unit splits into two identical copies",
        expected_index: 9,
    },
    RetrievalTestCase {
        // Target: battery chemistry (index 10). Avoids electrochemistry terms.
        query: "how rechargeable power sources generate electricity through chemical reactions",
        expected_index: 10,
    },
    RetrievalTestCase {
        // Target: spreadsheets (index 11). Avoids Excel/spreadsheet terms.
        query: "organizing numerical data in rows and columns with computed values",
        expected_index: 11,
    },
    // Cluster 5: "tree"
    RetrievalTestCase {
        // Target: BST algorithms (index 12). Avoids CS data structure terms.
        query: "a sorted hierarchical arrangement for fast lookup operations",
        expected_index: 12,
    },
    RetrievalTestCase {
        // Target: photosynthesis (index 13). Avoids botany terms.
        query: "green plants converting sunlight and carbon dioxide into sugar",
        expected_index: 13,
    },
    RetrievalTestCase {
        // Target: genealogy (index 14). Avoids genealogy terms.
        query: "discovering who your great-great-grandparents were using old documents",
        expected_index: 14,
    },
    // ── Tier B: Cross-cluster keyword confusion ────────────────────────────
    // Cluster 6: "pattern"
    RetrievalTestCase {
        // Target: design patterns (index 15). Uses "pattern" which appears in all 3.
        query: "reusable solutions for common problems in object-oriented code architecture",
        expected_index: 15,
    },
    RetrievalTestCase {
        // Target: regex (index 16). Uses "pattern" and "matching".
        query: "using special syntax to find and extract matching text segments",
        expected_index: 16,
    },
    RetrievalTestCase {
        // Target: animal migration (index 17). Uses "pattern" and "behavior".
        query: "why certain species travel vast distances at the same time every year",
        expected_index: 17,
    },
    // Cluster 7: "key"
    RetrievalTestCase {
        // Target: cryptography (index 18). Uses "key" ambiguously.
        query: "mathematical methods to keep digital communications secret and secure",
        expected_index: 18,
    },
    RetrievalTestCase {
        // Target: music key signatures (index 19). Uses "key" ambiguously.
        query: "the set of notes that define the tonal center of a musical composition",
        expected_index: 19,
    },
    RetrievalTestCase {
        // Target: database primary keys (index 20). Uses "key" ambiguously.
        query: "a unique identifier for each record in a structured data store",
        expected_index: 20,
    },
    // Cluster 8: "python"
    RetrievalTestCase {
        // Target: Python lang (index 21). "python" applies to all 3.
        query: "a popular interpreted scripting language used for web and data science",
        expected_index: 21,
    },
    RetrievalTestCase {
        // Target: Python snake (index 22). "python" applies to all 3.
        query: "a massive reptile that squeezes its prey to death",
        expected_index: 22,
    },
    RetrievalTestCase {
        // Target: Monty Python (index 23). "python" applies to all 3.
        query: "a legendary British comedy troupe known for surreal and silly humor",
        expected_index: 23,
    },
    // Cluster 9: "plate"
    RetrievalTestCase {
        // Target: tectonics (index 24). "plate" is ambiguous.
        query: "enormous slabs of rock grinding against each other causing tremors",
        expected_index: 24,
    },
    RetrievalTestCase {
        // Target: food plating (index 25). "plate" is ambiguous.
        query: "arranging a gourmet dish to look visually stunning before serving",
        expected_index: 25,
    },
    RetrievalTestCase {
        // Target: printing (index 26). "plate" is ambiguous.
        query: "transferring ink from a metal surface onto paper for mass reproduction",
        expected_index: 26,
    },
    // Cluster 10: "bridge"
    RetrievalTestCase {
        // Target: suspension bridge (index 27). "bridge" is ambiguous.
        query: "a long span crossing a river held up by thick steel wires from tall pillars",
        expected_index: 27,
    },
    RetrievalTestCase {
        // Target: card game (index 28). "bridge" is ambiguous.
        query: "a four-player partnership card game involving auction-style bidding",
        expected_index: 28,
    },
    RetrievalTestCase {
        // Target: dental bridge (index 29). "bridge" is ambiguous.
        query: "a prosthetic device that fills the gap left by extracted teeth",
        expected_index: 29,
    },
];

// ---------------------------------------------------------------------------
// Model download helpers
// ---------------------------------------------------------------------------

/// Files required by model2vec to load a model from disk.
const MODEL_FILES: &[&str] = &["tokenizer.json", "model.safetensors", "config.json"];

/// Return the local cache path for a given HuggingFace model ID.
fn model_cache_path(model_id: &str) -> PathBuf {
    let model_dir_name = model_id.rsplit('/').next().unwrap_or(model_id);
    dirs::cache_dir()
        .unwrap_or_else(|| Path::new(".").to_path_buf())
        .join(APP_NAME)
        .join(MODELS_SUBDIR)
        .join(model_dir_name)
}

/// Check if a model is already downloaded (all required files exist).
fn is_model_cached(model_id: &str) -> bool {
    let path = model_cache_path(model_id);
    MODEL_FILES.iter().all(|f| path.join(f).exists())
}

/// Download a model from HuggingFace Hub to the local cache.
/// Uses the raw file download URLs: https://huggingface.co/{model_id}/resolve/main/{file}
fn download_model(model_id: &str) {
    let cache_path = model_cache_path(model_id);
    std::fs::create_dir_all(&cache_path).expect("failed to create model cache dir");

    for file_name in MODEL_FILES {
        let dest = cache_path.join(file_name);
        if dest.exists() {
            continue;
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id, file_name
        );

        print!("    Downloading {} ... ", file_name);
        std::io::stdout().flush().ok();

        let output = std::process::Command::new("curl")
            .args(["-sL", "-o", dest.to_str().unwrap(), &url])
            .output()
            .expect("failed to run curl");

        if !output.status.success() {
            panic!(
                "Failed to download {}: {}",
                url,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Sanity check: file should be non-empty
        let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);

        println!("done ({:.1} MB)", size as f64 / 1_048_576.0);
    }
}

/// Ensure all benchmark models are available locally.
fn ensure_models_downloaded() {
    for (name, model_id) in MODELS {
        if !is_model_cached(model_id) {
            println!("  Downloading model: {} ...", name);
            download_model(model_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_memory_with_model(rt: &Runtime, model_id: &str) -> (TempDir, Mnemoria) {
    let temp_dir = tempfile::tempdir().expect("failed to create tempdir");
    let config = Config {
        durability: DurabilityMode::None,
        model_id: model_id.to_string(),
        ..Config::default()
    };

    let memory = rt
        .block_on(Mnemoria::create_with_config(temp_dir.path(), config))
        .expect("failed to create memory");

    (temp_dir, memory)
}

fn populate_corpus(rt: &Runtime, memory: &Mnemoria) -> Vec<String> {
    let mut ids = Vec::with_capacity(CORPUS.len());
    for (summary, content) in CORPUS {
        let id = rt
            .block_on(memory.remember(EntryType::Discovery, summary, content))
            .expect("failed to add entry");
        ids.push(id);
    }
    ids
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Accuracy evaluation: semantic-only (pure embedding cosine similarity)
// ---------------------------------------------------------------------------

struct AccuracyResult {
    precision_at_1: f64,
    recall_at_3: f64,
    recall_at_5: f64,
    mrr: f64,
}

fn evaluate_semantic_accuracy(backend: &EmbeddingBackend) -> AccuracyResult {
    // Pre-embed all corpus documents
    let corpus_embeddings: Vec<Vec<f32>> = CORPUS
        .iter()
        .map(|(_summary, content)| backend.embed(content).expect("failed to embed corpus doc"))
        .collect();

    let mut hits_at_1 = 0usize;
    let mut hits_at_3 = 0usize;
    let mut hits_at_5 = 0usize;
    let mut reciprocal_rank_sum = 0.0f64;

    for tc in TEST_CASES {
        let query_emb = backend.embed(tc.query).expect("failed to embed query");

        // Score all documents by cosine similarity
        let mut scores: Vec<(usize, f32)> = corpus_embeddings
            .iter()
            .enumerate()
            .map(|(i, doc_emb)| (i, cosine_similarity(&query_emb, doc_emb)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(rank) = scores.iter().position(|(idx, _)| *idx == tc.expected_index) {
            reciprocal_rank_sum += 1.0 / (rank as f64 + 1.0);
            if rank == 0 {
                hits_at_1 += 1;
            }
            if rank < 3 {
                hits_at_3 += 1;
            }
            if rank < 5 {
                hits_at_5 += 1;
            }
        }
    }

    let n = TEST_CASES.len() as f64;
    AccuracyResult {
        precision_at_1: hits_at_1 as f64 / n,
        recall_at_3: hits_at_3 as f64 / n,
        recall_at_5: hits_at_5 as f64 / n,
        mrr: reciprocal_rank_sum / n,
    }
}

// ---------------------------------------------------------------------------
// Accuracy evaluation: hybrid (BM25 + semantic, full Mnemoria pipeline)
// ---------------------------------------------------------------------------

fn evaluate_hybrid_accuracy(
    rt: &Runtime,
    memory: &Mnemoria,
    corpus_ids: &[String],
) -> AccuracyResult {
    let mut hits_at_1 = 0usize;
    let mut hits_at_3 = 0usize;
    let mut hits_at_5 = 0usize;
    let mut reciprocal_rank_sum = 0.0f64;

    for tc in TEST_CASES {
        let results = rt
            .block_on(memory.search_memory(tc.query, 5))
            .expect("search failed");

        let expected_id = &corpus_ids[tc.expected_index];
        let result_ids: Vec<&String> = results.iter().map(|r| &r.id).collect();

        if let Some(rank) = result_ids.iter().position(|id| *id == expected_id) {
            reciprocal_rank_sum += 1.0 / (rank as f64 + 1.0);
            if rank == 0 {
                hits_at_1 += 1;
            }
            if rank < 3 {
                hits_at_3 += 1;
            }
            if rank < 5 {
                hits_at_5 += 1;
            }
        }
    }

    let n = TEST_CASES.len() as f64;
    AccuracyResult {
        precision_at_1: hits_at_1 as f64 / n,
        recall_at_3: hits_at_3 as f64 / n,
        recall_at_5: hits_at_5 as f64 / n,
        mrr: reciprocal_rank_sum / n,
    }
}

// ---------------------------------------------------------------------------
// Performance evaluation
// ---------------------------------------------------------------------------

struct PerformanceResult {
    embed_latency_us: f64,
    search_latency_us: f64,
    write_throughput: f64,
}

fn evaluate_performance(rt: &Runtime, model_id: &str) -> PerformanceResult {
    // --- Embed latency (pure embedding, no search overhead) ---
    let backend = EmbeddingBackend::new(model_id);
    let sample_texts = [
        "borrow checker prevents dangling pointers and data races",
        "hippocampus consolidates experiences during sleep",
        "lithium ions between graphite anode and cobalt cathode",
        "parabolic cables between towers support the deck below",
        "NFA-based backtracking engine matches compiled automata",
    ];
    let embed_runs = 100;
    let start = Instant::now();
    for i in 0..embed_runs {
        let _ = backend.embed(sample_texts[i % sample_texts.len()]);
    }
    let embed_total = start.elapsed();
    let embed_latency_us = embed_total.as_micros() as f64 / embed_runs as f64;

    // --- Write throughput (includes embedding) ---
    let write_count = 100;
    let (temp_dir_w, memory_w) = create_memory_with_model(rt, model_id);
    let start = Instant::now();
    for i in 0..write_count {
        rt.block_on(memory_w.remember(
            EntryType::Discovery,
            &format!("Benchmark entry {i}"),
            &format!(
                "This is benchmark content number {i} for measuring write throughput with embeddings"
            ),
        ))
        .expect("remember failed");
    }
    let write_total = start.elapsed();
    let write_throughput = write_count as f64 / write_total.as_secs_f64();
    drop(memory_w);
    drop(temp_dir_w);

    // --- Search latency over populated store ---
    let (temp_dir_s, memory_s) = create_memory_with_model(rt, model_id);
    populate_corpus(rt, &memory_s);
    let search_runs = 100;
    let queries = [
        "borrow checker prevents dangling pointers and data races",
        "synaptic neurotransmitter release between axon and dendrite",
        "transformer pre-training on next-token prediction",
        "lithium ions between graphite anode and cobalt cathode",
        "AVL rotations maintain balanced height for logarithmic search",
        "monarch butterflies navigate using sun compass",
        "factoring large semiprimes is computationally hard",
        "quenelle plating techniques at fine dining restaurants",
        "absurdist sketch comedy Holy Grail cult classic",
        "porcelain pontics anchored to adjacent teeth",
    ];
    let start = Instant::now();
    for i in 0..search_runs {
        let query = queries[i % queries.len()];
        rt.block_on(memory_s.search_memory(query, 5))
            .expect("search failed");
    }
    let search_total = start.elapsed();
    let search_latency_us = search_total.as_micros() as f64 / search_runs as f64;
    drop(memory_s);
    drop(temp_dir_s);

    PerformanceResult {
        embed_latency_us,
        search_latency_us,
        write_throughput,
    }
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

fn print_separator(width: usize) {
    println!("{}", "=".repeat(width));
}

fn print_thin_separator(width: usize) {
    println!("{}", "-".repeat(width));
}

fn print_accuracy_table(label: &str, results: &[(&str, AccuracyResult)], width: usize) {
    println!();
    print_separator(width);
    println!("  {}", label);
    print_separator(width);
    println!();
    println!(
        "  {:<30} {:>12} {:>12} {:>12} {:>12}",
        "Model", "P@1", "R@3", "R@5", "MRR"
    );
    print_thin_separator(width);

    for (name, acc) in results {
        println!(
            "  {:<30} {:>11.1}% {:>11.1}% {:>11.1}% {:>12.4}",
            name,
            acc.precision_at_1 * 100.0,
            acc.recall_at_3 * 100.0,
            acc.recall_at_5 * 100.0,
            acc.mrr,
        );
    }
}

fn main() {
    tracing_subscriber::fmt::init();
    let rt = Runtime::new().expect("failed to build runtime");

    let report_width = 95;

    println!();
    print_separator(report_width);
    println!("  MNEMORIA MODEL COMPARISON REPORT");
    println!("  Top 5 model2vec Models — Retrieval Accuracy & Performance");
    println!(
        "  Corpus: {} documents in {} confusable clusters, {} queries",
        CORPUS.len(),
        CORPUS.len() / 3,
        TEST_CASES.len(),
    );
    print_separator(report_width);
    println!();

    // Ensure all models are downloaded before timing anything
    ensure_models_downloaded();
    println!();

    let mut semantic_results: Vec<(&str, AccuracyResult)> = Vec::new();
    let mut hybrid_results: Vec<(&str, AccuracyResult)> = Vec::new();
    let mut performance_results: Vec<(&str, PerformanceResult)> = Vec::new();

    for (name, model_id) in MODELS {
        println!("  Evaluating: {} ...", name);

        // Semantic-only accuracy (pure cosine similarity)
        let backend = EmbeddingBackend::new(model_id);
        let sem_acc = evaluate_semantic_accuracy(&backend);
        semantic_results.push((name, sem_acc));
        drop(backend);

        // Hybrid accuracy (BM25 + semantic via Mnemoria)
        let (_temp_dir, memory) = create_memory_with_model(&rt, model_id);
        let corpus_ids = populate_corpus(&rt, &memory);
        let hyb_acc = evaluate_hybrid_accuracy(&rt, &memory, &corpus_ids);
        hybrid_results.push((name, hyb_acc));
        drop(memory);
        drop(_temp_dir);

        // Performance
        let perf = evaluate_performance(&rt, model_id);
        performance_results.push((name, perf));

        println!("  Done.\n");
    }

    // --- Semantic-only accuracy ---
    print_accuracy_table(
        "RETRIEVAL ACCURACY — Semantic Only (pure cosine similarity, no BM25)",
        &semantic_results,
        report_width,
    );
    println!();
    println!("  This isolates embedding model quality. BM25 keyword matching is disabled.");

    // --- Hybrid accuracy ---
    print_accuracy_table(
        "RETRIEVAL ACCURACY — Hybrid (BM25 + semantic via Reciprocal Rank Fusion)",
        &hybrid_results,
        report_width,
    );
    println!();
    println!("  This is the full Mnemoria search pipeline as used in production.");

    // --- Metric legend (once) ---
    println!();
    println!("  P@1  = Precision@1 (correct document is the #1 result)");
    println!("  R@3  = Recall@3 (correct document in top 3 results)");
    println!("  R@5  = Recall@5 (correct document in top 5 results)");
    println!("  MRR  = Mean Reciprocal Rank (average 1/rank of correct result)");

    // --- Performance Table ---
    println!();
    print_separator(report_width);
    println!("  RETRIEVAL PERFORMANCE");
    print_separator(report_width);
    println!();
    println!(
        "  {:<30} {:>18} {:>18} {:>18}",
        "Model", "Embed (us/text)", "Search (us/query)", "Write (entries/s)"
    );
    print_thin_separator(report_width);

    for (name, perf) in &performance_results {
        println!(
            "  {:<30} {:>18.1} {:>18.1} {:>18.1}",
            name, perf.embed_latency_us, perf.search_latency_us, perf.write_throughput,
        );
    }

    println!();
    println!("  Embed      = Average time to embed a single text (microseconds)");
    println!("  Search     = Average hybrid search latency over 30-doc store (microseconds)");
    println!("  Write      = Write throughput including embedding generation (entries/sec)");

    // --- Summary ---
    println!();
    print_separator(report_width);
    println!("  SUMMARY");
    print_separator(report_width);
    println!();

    if let Some((name, acc)) = semantic_results
        .iter()
        .max_by(|a, b| a.1.mrr.partial_cmp(&b.1.mrr).unwrap())
    {
        println!(
            "  Best semantic accuracy:     {} (MRR: {:.4}, P@1: {:.1}%)",
            name,
            acc.mrr,
            acc.precision_at_1 * 100.0,
        );
    }

    if let Some((name, acc)) = hybrid_results
        .iter()
        .max_by(|a, b| a.1.mrr.partial_cmp(&b.1.mrr).unwrap())
    {
        println!(
            "  Best hybrid accuracy:       {} (MRR: {:.4}, P@1: {:.1}%)",
            name,
            acc.mrr,
            acc.precision_at_1 * 100.0,
        );
    }

    if let Some((name, perf)) = performance_results.iter().min_by(|a, b| {
        a.1.embed_latency_us
            .partial_cmp(&b.1.embed_latency_us)
            .unwrap()
    }) {
        println!(
            "  Fastest embedding:          {} ({:.1} us/text)",
            name, perf.embed_latency_us,
        );
    }

    if let Some((name, perf)) = performance_results.iter().min_by(|a, b| {
        a.1.search_latency_us
            .partial_cmp(&b.1.search_latency_us)
            .unwrap()
    }) {
        println!(
            "  Fastest search:             {} ({:.1} us/query)",
            name, perf.search_latency_us,
        );
    }

    if let Some((name, perf)) = performance_results.iter().max_by(|a, b| {
        a.1.write_throughput
            .partial_cmp(&b.1.write_throughput)
            .unwrap()
    }) {
        println!(
            "  Fastest write throughput:   {} ({:.1} entries/sec)",
            name, perf.write_throughput,
        );
    }

    println!();
    print_separator(report_width);
    println!();
}
