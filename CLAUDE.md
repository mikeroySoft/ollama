# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ollama is an open-source tool for running large language models locally. It provides a simple API for creating, managing, and running models, along with a library of pre-built models. The project is built in Go and integrates with llama.cpp for model inference.

## Development Commands

### Building and Running
- `go run . serve` - Start the Ollama server locally
- `cmake -B build && cmake --build build` - Build with CMake (required on macOS Intel, Windows, Linux)
- `go build` - Build the ollama binary

### Testing
- `go test ./...` - Run all tests
- `GOEXPERIMENT=synctest go test ./...` - Run tests with synctest package (recommended for CI compatibility)

### Model Management
- `go run . pull <model>` - Download a model (e.g., `go run . pull gemma3`)
- `go run . run <model>` - Run and chat with a model
- `go run . list` - List installed models
- `go run . ps` - Show currently loaded models
- `go run . stop <model>` - Stop a running model
- `go run . export <model> <path> [--compress]` - Export a model to file/directory
- `go run . import <path>` - Import a model from export

### Upstream Sync (Maintainers)
- `make -f Makefile.sync sync` - Sync with upstream llama.cpp repository
- `make -f Makefile.sync clean` - Clean local repository

## Architecture Overview

### Core Components

**Server Layer (`server/`)**
- `routes.go` - HTTP API endpoints including OpenAI-compatible API
- `sched.go` - Model loading/unloading scheduler
- `download.go` - Model downloading and registry operations
- REST API serves at `http://localhost:11434` by default

**LLM Management (`llm/`)**
- `server.go` - LLM server process management and communication
- Platform-specific implementations (`llm_darwin.go`, `llm_linux.go`, `llm_windows.go`)
- Memory management and GPU detection

**Model Handling (`model/`)**
- Text processing, tokenization, and vocabulary management
- Support for multiple model architectures in `models/` subdirectory
- Image processing for multimodal models in `imageproc/`

**Convert System (`convert/`)**
- Converts models from various formats (SafeTensors, PyTorch) to GGUF
- Model-specific converters (Llama, Gemma, Mistral, etc.)
- Tensor operations and quantization

**Command Line (`cmd/`)**
- CLI interface using Cobra framework
- Interactive chat, model management commands
- Progress indicators and formatting

**Backend Integration (`ml/backend/ggml/`)**
- Integration with ggml/llama.cpp for model inference
- GPU acceleration support (CUDA, ROCm, Metal)
- Quantization and optimization

### Key Flows

**Model Loading**: CLI/API → Server Router → Scheduler → LLM Server → llama.cpp backend
**Model Download**: CLI/API → Download Manager → Registry Client → Blob Storage
**Inference**: API Request → Server → LLM Server → Backend → Response Streaming

### Configuration

Environment variables are handled in `envconfig/config.go`:
- `OLLAMA_HOST` - Server host/port
- `OLLAMA_MODELS` - Model storage directory  
- `OLLAMA_NUM_PARALLEL` - Concurrent model limit
- GPU-specific variables (`CUDA_VISIBLE_DEVICES`, etc.)
- `OLLAMA_EXPORT_WORKERS` - Number of parallel workers for export (1-16, default: 4)

### Template System

Chat templates in `template/` define how conversations are formatted for different models. Each model family has its own template (e.g., `llama3-instruct.gotmpl`, `gemma-instruct.gotmpl`).

## Development Notes

- The project uses Go modules - ensure you're in the project root when running go commands
- GPU support requires platform-specific build steps (see `docs/development.md`)
- Model files are stored in GGUF format, converted from original formats as needed
- The server can run multiple models concurrently based on available memory
- OpenAI-compatible API endpoints are available at `/v1/` for integration
- Docker builds are supported with optional GPU acceleration

### Export/Import Performance Optimizations

The export functionality has been optimized for large models:

**Export (`server/export.go`)**
- Parallel blob reading with configurable workers (via `OLLAMA_EXPORT_WORKERS`)
- 64MB buffer sizes for file I/O operations (up from 1MB)
- Throttled progress updates to reduce overhead
- Uncompressed tar exports use `exportToTarParallel()` for best performance
- Alternative streaming export available in `export_streaming.go` for memory-constrained systems

**Import (`server/import.go`)**
- Currently uses 1MB buffers (can be optimized similarly)
- Tar imports extract to temp files then copy (can be optimized to stream directly)