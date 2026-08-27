/// Backend for computing text embeddings using model2vec.
///
/// Wraps an optional [`model2vec::Model2Vec`] model. When the `model2vec`
/// feature is disabled or the model fails to load, the backend gracefully
/// degrades — [`is_available`](Self::is_available) returns `false` and
/// [`embed`](Self::embed) returns an error. This allows the rest of the
/// system to fall back to BM25-only search.
pub struct EmbeddingBackend {
    #[cfg(feature = "model2vec")]
    model: Option<model2vec::Model2Vec>,
    #[cfg(not(feature = "model2vec"))]
    model: Option<()>,
}

impl EmbeddingBackend {
    /// Create a new embedding backend using the given HuggingFace model ID.
    ///
    /// The `model_id` should be a HuggingFace model identifier such as
    /// `"sentence-transformers/static-retrieval-mrl-en-v1"` (the default),
    /// `"minishlab/potion-base-32M"`, or any other model2vec-compatible model.
    /// Sentence-transformers static-embedding layouts are auto-converted.
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
            // Some public static-embedding models (e.g. the retrieval-tuned
            // sentence-transformers/static-retrieval-mrl-en-v1) ship in the
            // nested sentence-transformers layout:
            //   snapshots/<hash>/0_StaticEmbedding/{model.safetensors, tokenizer.json}
            // with a `embedding.weight` tensor. The stock model2vec loader
            // expects a flat layout (`embeddings` tensor + root config.json),
            // so materialize the flat layout once, next to the snapshot.
            Self::materialize_flat_layout(&model_path);

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

    /// Materialize a flat Model2Vec layout from a sentence-transformers
    /// static-embedding snapshot, when needed.
    ///
    /// Public static models (such as the retrieval-tuned
    /// `sentence-transformers/static-retrieval-mrl-en-v1`) are published in a
    /// nested layout:
    ///
    /// ```text
    /// <snapshot>/0_StaticEmbedding/model.safetensors   (tensor: embedding.weight)
    /// <snapshot>/0_StaticEmbedding/tokenizer.json
    /// ```
    ///
    /// The stock model2vec loader needs a flat layout at the snapshot root:
    /// `model.safetensors` with an `embeddings` tensor, plus `tokenizer.json`
    /// and `config.json`. This helper writes those flat files beside the
    /// nested ones the first time a model is loaded. It is idempotent and
    /// no-ops for models that already have a flat layout.
    #[cfg(feature = "model2vec")]
    fn materialize_flat_layout(model_path: &std::path::Path) {
        // Already flat (the common case for minishlab potion models).
        if model_path.join("model.safetensors").exists() && model_path.join("config.json").exists()
        {
            return;
        }

        let nested_dir = model_path.join("0_StaticEmbedding");
        let nested_safetensors = nested_dir.join("model.safetensors");
        let nested_tokenizer = nested_dir.join("tokenizer.json");
        if !nested_safetensors.exists() {
            return;
        }

        // Read the nested safetensors header and locate the embedding tensor.
        let Ok(bytes) = std::fs::read(&nested_safetensors) else {
            return;
        };
        if bytes.len() < 8 {
            return;
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if 8 + header_len > bytes.len() {
            return;
        }
        let Ok(header_json): Result<serde_json::Value, _> =
            serde_json::from_slice(&bytes[8..8 + header_len])
        else {
            return;
        };
        let Some(header_obj) = header_json.as_object() else {
            return;
        };

        // Find the tensor to rename (embedding.weight -> embeddings).
        let tensor_name = header_obj
            .keys()
            .find(|k| k.as_str() == "embedding.weight")
            .cloned()
            .or_else(|| header_obj.keys().next().cloned());
        let Some(tensor_name) = tensor_name else {
            return;
        };

        let Some(tensor_meta) = header_obj.get(&tensor_name).and_then(|v| v.as_object()) else {
            return;
        };
        let Some(dtype) = tensor_meta.get("dtype").and_then(|v| v.as_str()) else {
            return;
        };
        let Some(shape) = tensor_meta.get("shape").and_then(|v| v.as_array()) else {
            return;
        };
        let Some(offsets) = tensor_meta.get("data_offsets").and_then(|v| v.as_array()) else {
            return;
        };
        let (Some(start), Some(end)) = (
            offsets.first().and_then(|v| v.as_u64()),
            offsets.get(1).and_then(|v| v.as_u64()),
        ) else {
            return;
        };
        let start = start as usize;
        let end = end as usize;
        let data_start = 8 + header_len;
        if start > end || data_start + end > bytes.len() {
            return;
        }

        // Build the flat header with the renamed tensor.
        let mut flat_header = serde_json::Map::new();
        let mut flat_meta = serde_json::Map::new();
        flat_meta.insert(
            "dtype".to_string(),
            serde_json::Value::String(dtype.to_string()),
        );
        flat_meta.insert("shape".to_string(), serde_json::Value::Array(shape.clone()));
        flat_meta.insert(
            "data_offsets".to_string(),
            serde_json::json!([0u64, (end - start) as u64]),
        );
        flat_header.insert(
            "embeddings".to_string(),
            serde_json::Value::Object(flat_meta),
        );
        let flat_header_bytes = match serde_json::to_vec(&serde_json::Value::Object(flat_header)) {
            Ok(b) => b,
            Err(_) => return,
        };

        // Assemble the flat file: header length + header + tensor data.
        let tensor_data = &bytes[data_start + start..data_start + end];
        let mut flat_bytes = Vec::with_capacity(8 + flat_header_bytes.len() + tensor_data.len());
        flat_bytes.extend_from_slice(&(flat_header_bytes.len() as u64).to_le_bytes());
        flat_bytes.extend_from_slice(&flat_header_bytes);
        flat_bytes.extend_from_slice(tensor_data);

        if std::fs::write(model_path.join("model.safetensors"), flat_bytes).is_err() {
            return;
        }

        // Copy the tokenizer up to the root and write a minimal config.json.
        if nested_tokenizer.exists() {
            let _ = std::fs::copy(&nested_tokenizer, model_path.join("tokenizer.json"));
        }
        let _ = std::fs::write(
            model_path.join("config.json"),
            serde_json::json!({"normalize": true}).to_string(),
        );

        tracing::info!(
            "Materialized flat Model2Vec layout from nested static-embedding snapshot at {}",
            model_path.display(),
        );
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

        // Accept the snapshot if it has the flat Model2Vec files, or the
        // nested sentence-transformers static layout that materialize_flat_layout
        // can convert.
        let flat_ok = snapshot_dir.join("tokenizer.json").exists()
            && snapshot_dir.join("model.safetensors").exists();
        let nested_ok = snapshot_dir
            .join("0_StaticEmbedding")
            .join("model.safetensors")
            .exists();
        if flat_ok || nested_ok {
            Some(snapshot_dir)
        } else {
            None
        }
    }

    /// Compute the embedding vector for the given text.
    ///
    /// Returns a `Vec<f32>` whose dimensionality depends on the loaded model
    /// (1024 for the default `static-retrieval-mrl-en-v1`, 512 for the potion models).
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
