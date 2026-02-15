/// Application name, used as the default directory name for memory stores,
/// cache directories, and ephemeral index prefixes.
pub const APP_NAME: &str = "mnemoria";

/// Subdirectory name for cached embedding models under the app cache directory.
pub const MODELS_SUBDIR: &str = "models";

/// Default model2vec model ID used when none is specified.
pub const DEFAULT_MODEL_ID: &str = "minishlab/potion-base-8M";

/// Filename for the manifest JSON file within a memory store directory.
pub(crate) const MANIFEST_FILENAME: &str = "manifest.json";

/// Filename for the append-only binary log within a memory store directory.
pub(crate) const LOG_FILENAME: &str = "log.bin";
