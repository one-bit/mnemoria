/// Backend for computing text embeddings using model2vec.
///
/// Wraps an optional [`model2vec::Model2Vec`] model. When the `model2vec`
/// feature is disabled or the model fails to load, the backend gracefully
/// degrades — [`is_available`](Self::is_available) returns `false` and
/// [`embed`](Self::embed) returns an error. This allows the rest of the
/// system to fall back to BM25-only search.
pub struct EmbeddingBackend {
    model: Option<model2vec::Model2Vec>,
}

impl EmbeddingBackend {
    /// Create a new embedding backend using the given HuggingFace model ID.
    ///
    /// The `model_id` should be a HuggingFace model identifier such as
    /// `"minishlab/potion-base-8M"` or `"minishlab/potion-base-32M"`.
    /// Any model2vec-compatible model can be used.
    pub fn new(model_id: &str) -> Self {
        #[cfg(feature = "model2vec")]
        {
            match Self::try_load_model(model_id) {
                Ok(model) => Self { model: Some(model) },
                Err(e) => {
                    tracing::warn!(
                        "Failed to load embedding model '{}': {}. Embeddings will be disabled.",
                        model_id,
                        e
                    );
                    Self { model: None }
                }
            }
        }

        #[cfg(not(feature = "model2vec"))]
        {
            let _ = model_id;
            Self { model: None }
        }
    }

    #[cfg(feature = "model2vec")]
    fn try_load_model(model_id: &str) -> Result<model2vec::Model2Vec, crate::Error> {
        use std::path::Path;

        // If the model_id is already an absolute path to a directory, try it directly.
        let as_path = Path::new(model_id);
        if as_path.is_absolute() && as_path.is_dir() {
            return model2vec::Model2Vec::from_pretrained(as_path, None, None)
                .map_err(|e: anyhow::Error| crate::Error::Embedding(e.to_string()));
        }

        // Look up the model in the HuggingFace cache.
        // HF cache layout: <cache>/huggingface/hub/models--<org>--<name>/snapshots/<hash>/
        if let Some(model_path) = Self::resolve_hf_cache_path(model_id) {
            match model2vec::Model2Vec::from_pretrained(&model_path, None, None) {
                Ok(model) => return Ok(model),
                Err(e) => {
                    tracing::warn!(
                        "Found model in HuggingFace cache at {} but failed to load: {}",
                        model_path.display(),
                        e
                    );
                }
            }
        }

        Err(crate::Error::Embedding(format!(
            "Model '{}' not found in HuggingFace cache (~/.cache/huggingface/hub/). \
             Download it first with: pip install huggingface_hub && \
             huggingface-cli download {}",
            model_id, model_id
        )))
    }

    /// Resolve a HuggingFace model ID to a local snapshot path in the HF cache.
    ///
    /// The HuggingFace cache layout is:
    /// ```text
    /// <cache_dir>/huggingface/hub/models--<org>--<name>/
    ///   refs/main         -> contains the commit hash
    ///   snapshots/<hash>/ -> tokenizer.json, model.safetensors, config.json
    /// ```
    ///
    /// Respects the `HF_HOME` and `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE`
    /// environment variables, falling back to `~/.cache/huggingface/hub`.
    #[cfg(feature = "model2vec")]
    fn resolve_hf_cache_path(model_id: &str) -> Option<std::path::PathBuf> {
        use std::path::PathBuf;

        // Determine the HF hub cache directory, respecting env vars.
        let hub_cache = if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
            PathBuf::from(dir)
        } else if let Ok(dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
            PathBuf::from(dir)
        } else if let Ok(hf_home) = std::env::var("HF_HOME") {
            PathBuf::from(hf_home).join("hub")
        } else {
            dirs::cache_dir()?.join("huggingface").join("hub")
        };

        // HF encodes model IDs as "models--<org>--<name>"
        let hf_dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_dir = hub_cache.join(&hf_dir_name);

        if !model_dir.exists() {
            return None;
        }

        // Read the commit hash from refs/main
        let refs_main = model_dir.join("refs").join("main");
        let commit_hash = std::fs::read_to_string(&refs_main).ok()?;
        let commit_hash = commit_hash.trim();

        let snapshot_dir = model_dir.join("snapshots").join(commit_hash);

        // Verify the snapshot directory has the required files
        if snapshot_dir.join("tokenizer.json").exists()
            && snapshot_dir.join("model.safetensors").exists()
        {
            Some(snapshot_dir)
        } else {
            None
        }
    }

    /// Compute the embedding vector for the given text.
    ///
    /// Returns a `Vec<f32>` whose dimensionality depends on the loaded model
    /// (256 for the default `potion-base-8M`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embedding`](crate::Error::Embedding) if the model is
    /// not loaded or inference fails.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, crate::Error> {
        #[cfg(feature = "model2vec")]
        {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| crate::Error::Embedding("Model not loaded".to_string()))?;

            let embeddings = model
                .encode([text])
                .map_err(|e: anyhow::Error| crate::Error::Embedding(e.to_string()))?;

            let embedding_vec = embeddings.row(0).to_vec();
            Ok(embedding_vec)
        }

        #[cfg(not(feature = "model2vec"))]
        {
            let _ = text;
            Err(crate::Error::Embedding(
                "model2vec feature not enabled".to_string(),
            ))
        }
    }

    /// Returns `true` if the embedding model is loaded and ready.
    pub fn is_available(&self) -> bool {
        self.model.is_some()
    }
}
