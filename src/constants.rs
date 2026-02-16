/// Application name, used as the default directory name for memory stores,
/// cache directories, and ephemeral index prefixes.
pub const APP_NAME: &str = "mnemoria";

/// Default model2vec model ID used when none is specified.
pub const DEFAULT_MODEL_ID: &str = "minishlab/potion-base-32M";

/// Filename for the manifest JSON file within a memory store directory.
pub(crate) const MANIFEST_FILENAME: &str = "manifest.json";

/// Filename for the append-only binary log within a memory store directory.
pub(crate) const LOG_FILENAME: &str = "log.bin";
