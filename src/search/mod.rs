use crate::types::MemoryEntry;
use simsimd::SpatialSimilarity;
use std::collections::HashMap;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, Field, STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, IndexReader, IndexWriter, TantivyDocument, TantivyError};

const RRF_K: f32 = 60.0;

/// Heap size in bytes allocated to the Tantivy IndexWriter (50 MB).
const TANTIVY_WRITER_HEAP_BYTES: usize = 50_000_000;

// Tantivy schema field names.
const FIELD_ID: &str = "id";
const FIELD_SUMMARY: &str = "summary";
const FIELD_CONTENT: &str = "content";
const FIELD_ENTRY_TYPE: &str = "entry_type";
const FIELD_TIMESTAMP: &str = "timestamp";

/// Manages a Tantivy full-text search index with an in-memory vector store
/// for hybrid BM25 + semantic search.
///
/// Each `IndexManager` owns an exclusive Tantivy [`IndexWriter`], so every
/// process must use its own index directory. The index is ephemeral and
/// rebuilt from the binary log on each [`Mnemoria::open`](crate::Mnemoria::open).
pub struct IndexManager {
    index: Index,
    reader: IndexReader,
    writer: Option<IndexWriter>,
    // Cached field handles (resolved once in new())
    id_field: Field,
    summary_field: Field,
    content_field: Field,
    entry_type_field: Field,
    timestamp_field: Field,
    vector_store: HashMap<String, Vec<f32>>,
}

impl IndexManager {
    /// Release the Tantivy IndexWriter so its file handles are closed.
    /// Called during Drop before removing the per-process index directory.
    pub fn drop_writer(&mut self) {
        self.writer = None;
    }

    pub fn new(index_path: &Path) -> Result<Self, crate::Error> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field(FIELD_ID, STRING | STORED);
        schema_builder.add_text_field(FIELD_SUMMARY, TEXT | STORED);
        schema_builder.add_text_field(FIELD_CONTENT, TEXT | STORED);
        schema_builder.add_text_field(FIELD_ENTRY_TYPE, STRING | STORED);
        schema_builder.add_i64_field(FIELD_TIMESTAMP, FAST | STORED);
        let schema = schema_builder.build();

        let index = if index_path.exists() {
            Index::open_in_dir(index_path)?
        } else {
            std::fs::create_dir_all(index_path)?;
            Index::create_in_dir(index_path, schema.clone())?
        };

        let reader = index.reader()?;
        let writer = index.writer(TANTIVY_WRITER_HEAP_BYTES)?;

        // Resolve field handles once — safe because we just built the schema above.
        let id_field = schema.get_field(FIELD_ID).expect("schema missing id field");
        let summary_field = schema
            .get_field(FIELD_SUMMARY)
            .expect("schema missing summary field");
        let content_field = schema
            .get_field(FIELD_CONTENT)
            .expect("schema missing content field");
        let entry_type_field = schema
            .get_field(FIELD_ENTRY_TYPE)
            .expect("schema missing entry_type field");
        let timestamp_field = schema
            .get_field(FIELD_TIMESTAMP)
            .expect("schema missing timestamp field");

        Ok(Self {
            index,
            reader,
            writer: Some(writer),
            id_field,
            summary_field,
            content_field,
            entry_type_field,
            timestamp_field,
            vector_store: HashMap::new(),
        })
    }

    pub fn add_entry(&mut self, entry: &MemoryEntry) -> Result<(), crate::Error> {
        let writer = self.writer.as_mut().ok_or_else(|| {
            crate::Error::Search(TantivyError::SystemError(
                "Writer not available".to_string(),
            ))
        })?;

        let mut doc = TantivyDocument::new();
        doc.add_text(self.id_field, &entry.id);
        doc.add_text(self.summary_field, &entry.summary);
        doc.add_text(self.content_field, &entry.content);
        doc.add_text(self.entry_type_field, entry.entry_type.to_string());
        doc.add_i64(self.timestamp_field, entry.timestamp);

        writer.add_document(doc)?;

        if let Some(ref embedding) = entry.embedding {
            self.vector_store
                .insert(entry.id.clone(), embedding.clone());
        }

        Ok(())
    }

    pub fn commit(&mut self) -> Result<(), crate::Error> {
        if let Some(writer) = self.writer.as_mut() {
            writer.commit()?;
            self.reader.reload()?;
        }
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>, crate::Error> {
        let searcher = self.reader.searcher();

        let query_parser =
            QueryParser::for_index(&self.index, vec![self.summary_field, self.content_field]);
        // Use lenient parsing so that special characters in user queries
        // (e.g. `c++`, unmatched quotes) don't cause hard errors.
        let (query, _parse_errors) = query_parser.parse_query_lenient(query);

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit * 2))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            if let Some(id) = doc.get_first(self.id_field)
                && let Some(text) = id.as_str()
            {
                results.push((text.to_string(), score));
            }
        }

        results.truncate(limit);
        Ok(results)
    }

    pub fn hybrid_search(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>, crate::Error> {
        let searcher = self.reader.searcher();

        let query_parser =
            QueryParser::for_index(&self.index, vec![self.summary_field, self.content_field]);
        // Use lenient parsing so that special characters in user queries
        // (e.g. `c++`, unmatched quotes) don't cause hard errors.
        let (parsed_query, _parse_errors) = query_parser.parse_query_lenient(query);

        let bm25_results: Vec<(String, f32)> = {
            let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(limit * 2))?;

            let mut results = Vec::new();
            for (score, doc_address) in top_docs {
                let doc: TantivyDocument = searcher.doc(doc_address)?;

                if let Some(id) = doc.get_first(self.id_field)
                    && let Some(text) = id.as_str()
                {
                    results.push((text.to_string(), score));
                }
            }
            results
        };

        if let Some(query_vec) = query_embedding {
            let vector_results: Vec<(String, f32)> = {
                let mut scores: Vec<(String, f32)> = Vec::new();

                for (id, doc_vector) in &self.vector_store {
                    if let Some(sim_score) = compute_cosine_similarity(query_vec, doc_vector) {
                        scores.push((id.clone(), sim_score));
                    }
                }

                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scores.into_iter().take(limit * 2).collect()
            };

            let combined = rrf_merge(&bm25_results, &vector_results, RRF_K);
            Ok(combined.into_iter().take(limit).collect())
        } else {
            Ok(bm25_results.into_iter().take(limit).collect())
        }
    }

    pub fn rebuild_from_entries(&mut self, entries: &[MemoryEntry]) -> Result<(), crate::Error> {
        if let Some(writer) = self.writer.as_mut() {
            writer.delete_all_documents()?;
            writer.commit()?;
        }

        self.vector_store.clear();

        for entry in entries {
            self.add_entry(entry)?;
        }

        self.commit()
    }

    pub fn clear(&mut self) -> Result<(), crate::Error> {
        if let Some(writer) = self.writer.as_mut() {
            writer.delete_all_documents()?;
            writer.commit()?;
            self.reader.reload()?;
        }
        self.vector_store.clear();
        Ok(())
    }
}

/// Compute cosine similarity using SIMD-accelerated simsimd.
///
/// simsimd returns cosine **distance** (1 - similarity), so we convert
/// back to similarity. Returns `None` for mismatched/empty/zero-norm vectors.
fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }

    // SpatialSimilarity::cos returns Option<f64> cosine distance
    let distance = <f32 as SpatialSimilarity>::cos(a, b)? as f32;
    let similarity = 1.0 - distance;
    Some(similarity)
}

fn rrf_merge(list1: &[(String, f32)], list2: &[(String, f32)], k: f32) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    let mut rrf_scores: HashMap<String, f32> = HashMap::new();

    for (rank, (id, _)) in list1.iter().enumerate() {
        let score = 1.0 / (k + rank as f32 + 1.0);
        *rrf_scores.entry(id.clone()).or_insert(0.0) += score;
    }

    for (rank, (id, _)) in list2.iter().enumerate() {
        let score = 1.0 / (k + rank as f32 + 1.0);
        *rrf_scores.entry(id.clone()).or_insert(0.0) += score;
    }

    let mut results: Vec<(String, f32)> = rrf_scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}
