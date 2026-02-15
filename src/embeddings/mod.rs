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

        // Derive a directory-safe name from the model ID.
        // e.g. "minishlab/potion-base-8M" -> "potion-base-8M"
        let model_dir_name = model_id.rsplit('/').next().unwrap_or(model_id);

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| Path::new(".").to_path_buf())
            .join(crate::constants::APP_NAME)
            .join(crate::constants::MODELS_SUBDIR);

        let model_path = cache_dir.join(model_dir_name);

        if model_path.exists() && model_path.join("tokenizer.json").exists() {
            match model2vec::Model2Vec::from_pretrained(&model_path, None, None) {
                Ok(model) => return Ok(model),
                Err(e) => {
                    tracing::warn!("Failed to load model from cache: {}", e);
                }
            }
        }

        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| crate::Error::Embedding(format!("Failed to create cache dir: {e}")))?;

        model2vec::Model2Vec::from_pretrained(model_id, None, None)
            .map_err(|e: anyhow::Error| crate::Error::Embedding(e.to_string()))
    }

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
            Err(crate::Error::Embedding(
                "model2vec feature not enabled".to_string(),
            ))
        }
    }

    pub fn is_available(&self) -> bool {
        self.model.is_some()
    }
}
