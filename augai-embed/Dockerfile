# Multi-stage build: compile in a full Rust image, then copy just the binary
# to a minimal Debian image. Final image ~150MB instead of ~2GB.

# Stage 1: Build the Rust binary
FROM rust:1.85-slim AS builder

WORKDIR /app
# pkg-config + libssl-dev needed by some transitive deps (reqwest → openssl)
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy manifests first for Docker layer caching — deps are rebuilt only when Cargo.toml changes
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/

ARG FEATURES
RUN cargo build --release $([ -n "$FEATURES" ] && echo "--features $FEATURES")

# Stage 2: Minimal runtime image (no Rust compiler, no build tools)
FROM debian:bookworm-slim

# ca-certificates: needed for HTTPS model downloads from HuggingFace on first run
# libssl3: runtime OpenSSL dependency
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

# Copy only the compiled binary from the builder stage (~50MB)
COPY --from=builder /app/target/release/augai-embed /usr/local/bin/augai-embed

ENV AUGAI_EMBED_PORT=8081
EXPOSE 8081

# CMD = default command when container starts (like "start" script in package.json)
CMD ["augai-embed"]
