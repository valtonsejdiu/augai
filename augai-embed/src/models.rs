// Model manager — lazy-loads ML models on first use (not at startup).
//
// Why lazy? Each model is 100-500MB. Loading all three at startup would mean
// a 10-30 second wait before the server accepts requests. Instead, each model
// loads on its first request (~2-5 seconds) and stays in memory forever after.
//
// Key Rust concepts used here:
// - OnceLock: thread-safe one-time initialization (like a lazy singleton).
//   Similar to `let instance; function get() { if (!instance) instance = new X(); return instance; }`
// - Arc: reference-counted smart pointer. Allows multiple owners of the same data
//   across threads. Think of it as a shared_ptr — the data lives until the last Arc is dropped.
// - Mutex: mutual exclusion lock. fastembed's embed() needs `&mut self` (exclusive access),
//   but we share the model across request handlers. Mutex ensures only one thread
//   calls embed() at a time. Like a JS mutex/semaphore for async code.

use std::sync::{Arc, Mutex, OnceLock};

use fastembed::{
    EmbeddingModel, InitOptions, SparseModel, SparseInitOptions, TextEmbedding,
    SparseTextEmbedding, TextRerank, RerankInitOptions, RerankerModel,
};

// Global singletons. `static` = lives for the entire program (like a global variable).
// OnceLock<T> = can only be written to once, then it's read-only forever.
static DENSE: OnceLock<Arc<Mutex<TextEmbedding>>> = OnceLock::new();
static SPARSE: OnceLock<Arc<Mutex<SparseTextEmbedding>>> = OnceLock::new();
static RERANKER: OnceLock<Arc<Mutex<TextRerank>>> = OnceLock::new();

// Returns the dense embedding model, loading it on first call.
// NomicEmbedTextV15: 768 dimensions, 8192 token context — 16x more context than the JS models.
pub fn dense() -> Arc<Mutex<TextEmbedding>> {
    DENSE
        .get_or_init(|| {
            tracing::info!("Loading dense model: NomicEmbedTextV15");
            // try_new() downloads the model from HuggingFace on first run (~270MB),
            // then loads the ONNX weights into memory. Subsequent runs use the cached files.
            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::NomicEmbedTextV15).with_show_download_progress(true),
            )
            .expect("Failed to load dense model");
            // Wrap in Arc<Mutex<>> so multiple request handlers can share it safely.
            // Arc = shared ownership, Mutex = exclusive access for embed() calls.
            Arc::new(Mutex::new(model))
        })
        .clone() // Clone the Arc (cheap — just increments a reference count, not the model itself)
}

// Returns the sparse embedding model (SPLADE).
// Produces sparse vectors: most dimensions are zero, only activated terms have values.
// Enables keyword-aware search that complements dense semantic search.
pub fn sparse() -> Arc<Mutex<SparseTextEmbedding>> {
    SPARSE
        .get_or_init(|| {
            tracing::info!("Loading sparse model: SPLADEPPV1");
            let model = SparseTextEmbedding::try_new(
                SparseInitOptions::new(SparseModel::SPLADEPPV1)
                    .with_show_download_progress(true),
            )
            .expect("Failed to load sparse model");
            Arc::new(Mutex::new(model))
        })
        .clone()
}

// Returns the cross-encoder reranker model.
// Unlike bi-encoders (dense/sparse) which embed query and doc separately,
// a cross-encoder processes (query, document) pairs together — more accurate
// but slower. Used as a second-stage ranker on the top-N candidates.
pub fn reranker() -> Arc<Mutex<TextRerank>> {
    RERANKER
        .get_or_init(|| {
            tracing::info!("Loading reranker model: BGERerankerV2M3");
            let model = TextRerank::try_new(
                RerankInitOptions::new(RerankerModel::BGERerankerV2M3)
                    .with_show_download_progress(true),
            )
            .expect("Failed to load reranker model");
            Arc::new(Mutex::new(model))
        })
        .clone()
}

// Returns a JSON summary of available models (for the GET /models endpoint).
pub fn available_models() -> serde_json::Value {
    serde_json::json!({
        "dense": ["NomicEmbedTextV15", "BGEBaseENV15"],
        "sparse": ["SPLADEPPV1"],
        "reranker": ["BGERerankerV2M3"]
    })
}
