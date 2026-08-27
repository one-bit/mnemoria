/// Application name, used as the default directory name for memory stores,
/// cache directories, and ephemeral index prefixes.
pub const APP_NAME: &str = "mnemoria";

/// Default model2vec model ID used when none is specified.
///
/// Since 0.4.1: retrieval-tuned static-retrieval-mrl-en-v1 (public, MIT-ish
/// Apache-2.0) — benchmark winner for semantic-only recall (MRR 0.9833 vs
/// 0.9667 for potion-base-32M) with identical hybrid MRR (0.8233) at the
/// same model size. The loader auto-materializes its nested static layout.
pub const DEFAULT_MODEL_ID: &str = "sentence-transformers/static-retrieval-mrl-en-v1";

/// Filename for the manifest JSON file within a memory store directory.
pub(crate) const MANIFEST_FILENAME: &str = "manifest.json";

/// Filename for the append-only binary log within a memory store directory.
pub(crate) const LOG_FILENAME: &str = "log.bin";
