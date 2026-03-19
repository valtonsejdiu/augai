// HTTP route handlers — each function handles one API endpoint.
//
// Key Rust/axum patterns:
// - `Json<T>`: axum extractor that auto-parses the request body as JSON into struct T.
//   Like `req.body` in Express with express.json() middleware, but type-safe at compile time.
// - `#[derive(Deserialize)]`: auto-generates JSON→struct parsing code (like Zod schemas but at compile time).
// - `#[derive(Serialize)]`: auto-generates struct→JSON serialization code.
// - `Result<impl IntoResponse, StatusCode>`: return type — either a success response or an HTTP error code.
//   The `?` operator (used after .map_err()) is Rust's equivalent of try/catch — it propagates errors up.
// - `impl IntoResponse`: means "any type that can become an HTTP response" (trait = interface in Rust).

use axum::{extract::Json, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};

use crate::models;

// ============================================================
// Dense embedding — POST /embed/dense
// Input:  { "texts": ["hello world", "another text"] }
// Output: { "vectors": [[0.1, 0.2, ...], [...]], "dimension": 768 }
// ============================================================

#[derive(Deserialize)]
pub struct DenseRequest {
    pub texts: Vec<String>,
}

#[derive(Serialize)]
pub struct DenseResponse {
    pub vectors: Vec<Vec<f32>>,  // f32 = 32-bit float (like Float32Array in JS)
    pub dimension: usize,        // usize = unsigned pointer-sized integer (like number in JS, but unsigned)
}

pub async fn embed_dense(Json(req): Json<DenseRequest>) -> Result<impl IntoResponse, StatusCode> {
    let model = models::dense();
    let texts = req.texts;

    // spawn_blocking() moves the closure to a dedicated thread pool for CPU-heavy work.
    // Without this, the synchronous fastembed computation would block the async runtime
    // and prevent other requests from being handled. Same concept as worker_threads in Node.js.
    //
    // `move ||` captures `model` and `texts` by value (moves ownership into the closure).
    // This is necessary because the closure runs on a different thread.
    let vectors = tokio::task::spawn_blocking(move || {
        // .lock().unwrap() acquires the mutex — blocks until no other thread is using the model.
        // .unwrap() panics if the mutex is poisoned (only happens if a previous holder panicked).
        let mut m = model.lock().unwrap();
        m.embed(texts, None)  // None = use default batch size
    })
    .await
    // First .map_err: handle spawn_blocking failure (thread pool panicked)
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    // Second .map_err: handle fastembed inference failure
    .map_err(|e| {
        tracing::error!("Dense embedding failed: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let dimension = vectors.first().map(|v| v.len()).unwrap_or(768);
    Ok(Json(DenseResponse { vectors, dimension }))
}

// ============================================================
// Sparse embedding — POST /embed/sparse
// Input:  { "texts": ["hello world"] }
// Output: { "vectors": [{ "indices": [42, 1337, ...], "values": [0.8, 0.3, ...] }] }
//
// Sparse vectors have mostly zero values. Only "activated" token positions
// have non-zero weights. SPLADE expands queries with related terms
// (e.g., "car" activates "vehicle", "automobile") — this catches keyword
// matches that dense embeddings sometimes miss.
// ============================================================

#[derive(Deserialize)]
pub struct SparseRequest {
    pub texts: Vec<String>,
}

#[derive(Serialize)]
pub struct SparseVector {
    pub indices: Vec<usize>,  // Which positions in the vocabulary are activated
    pub values: Vec<f32>,     // The weight/importance of each activated position
}

#[derive(Serialize)]
pub struct SparseResponse {
    pub vectors: Vec<SparseVector>,
}

pub async fn embed_sparse(
    Json(req): Json<SparseRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let model = models::sparse();
    let texts = req.texts;

    let embeddings = tokio::task::spawn_blocking(move || {
        let mut m = model.lock().unwrap();
        m.embed(texts, None)
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|e| {
        tracing::error!("Sparse embedding failed: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Convert fastembed's SparseEmbedding structs to our API response format.
    // .into_iter() consumes the Vec (takes ownership), .map() transforms each element,
    // .collect() gathers results into a new Vec. Like texts.map(x => ...) in JS.
    let vectors = embeddings
        .into_iter()
        .map(|emb| SparseVector {
            indices: emb.indices,
            values: emb.values,
        })
        .collect();

    Ok(Json(SparseResponse { vectors }))
}

// ============================================================
// Reranking — POST /rerank
// Input:  { "query": "database setup", "passages": ["pg setup...", "react hooks..."], "top_k": 3 }
// Output: { "results": [{ "index": 0, "score": 0.95 }, { "index": 1, "score": 0.12 }] }
//
// The reranker is a cross-encoder: it reads (query + passage) together and produces
// a relevance score. Much more accurate than cosine similarity because it can
// understand relationships between query and passage tokens (attention across both).
// Trade-off: ~10x slower per pair, so we only rerank the top candidates from retrieval.
// ============================================================

#[derive(Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub passages: Vec<String>,
    pub top_k: Option<usize>,   // Option<T> = T | undefined in TypeScript
}

#[derive(Serialize)]
pub struct RerankResultItem {
    pub index: usize,   // Original index in the passages array
    pub score: f32,     // Relevance score (higher = more relevant)
}

#[derive(Serialize)]
pub struct RerankResponse {
    pub results: Vec<RerankResultItem>,
}

pub async fn rerank(Json(req): Json<RerankRequest>) -> Result<impl IntoResponse, StatusCode> {
    let model = models::reranker();
    let query = req.query;
    let passages = req.passages;
    let _top_k = req.top_k;

    let results = tokio::task::spawn_blocking(move || {
        let mut m = model.lock().unwrap();
        // fastembed's rerank() expects &str slices, not Strings.
        // &str is a borrowed string slice (like a readonly view into a String).
        // We create temporary references that point into the owned Strings.
        let refs: Vec<&str> = passages.iter().map(|s| s.as_str()).collect();
        // Args: query, documents, return_documents (include text in results), batch_size
        m.rerank(query.as_str(), refs.as_slice(), false, None)
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|e| {
        tracing::error!("Reranking failed: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut items: Vec<RerankResultItem> = results
        .into_iter()
        .map(|r| RerankResultItem {
            index: r.index,
            score: r.score,
        })
        .collect();

    // Sort by score descending, then truncate to top_k if specified.
    // partial_cmp is needed because f32 can be NaN (NaN != NaN), so total ordering isn't guaranteed.
    items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = _top_k {
        items.truncate(k);  // Like items.slice(0, k) in JS, but in-place
    }

    Ok(Json(RerankResponse { results: items }))
}

// ============================================================
// Info endpoints
// ============================================================

// GET /models — lists available models per category
pub async fn models_info() -> impl IntoResponse {
    Json(models::available_models())
}

// GET /health — simple liveness check
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}
