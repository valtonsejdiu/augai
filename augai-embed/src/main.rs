// Declare submodules — Rust's module system (like import from "./models.js" but declared at crate level).
// Each `mod x;` tells the compiler to look for `src/x.rs` and include it in the binary.
mod models;
mod routes;

use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

// #[tokio::main] transforms `async fn main()` into a synchronous main() that
// boots up the tokio async runtime. Without this, Rust has no built-in async
// runtime (unlike Node.js which has one baked in via libuv).
#[tokio::main]
async fn main() {
    // Initialize structured logging.
    // Log level controlled by AUGAI_EMBED_LOG_LEVEL env var (e.g., "debug", "info", "warn").
    // Defaults to "info" if not set.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("AUGAI_EMBED_LOG_LEVEL")
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Read port from env, default to 8081.
    // The chain: env var → parse to u16 → fallback to 8081.
    // .ok() converts Result→Option (discards the error), .and_then() chains optionals.
    let port: u16 = std::env::var("AUGAI_EMBED_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8081);

    // Build the HTTP router — same concept as Express's app.get/app.post.
    // Each .route() maps a path + method to a handler function.
    // .layer() adds middleware — here, permissive CORS (allows any origin).
    let app = Router::new()
        .route("/embed/dense", post(routes::embed_dense))
        .route("/embed/sparse", post(routes::embed_sparse))
        .route("/rerank", post(routes::rerank))
        .route("/models", get(routes::models_info))
        .route("/health", get(routes::health))
        .layer(CorsLayer::permissive());

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("augai-embed listening on {addr}");
    tracing::info!("Available models: {}", models::available_models());

    // Bind a TCP listener and start serving.
    // .await in Rust is like `await` in JS — suspends until the future resolves.
    // .unwrap() panics on error (acceptable here — if we can't bind, we should crash).
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    // Serve the app with graceful shutdown — when Ctrl+C is pressed, finish
    // in-flight requests before exiting (like server.close() in Node.js).
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

// Returns a Future that resolves when Ctrl+C (SIGINT) is received.
// axum's graceful shutdown awaits this future — once it completes, the server
// stops accepting new connections and drains existing ones.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    tracing::info!("Shutting down...");
}
