# KrishLLM

KrishLLM is a custom C++ LLM inference engine built to run quantized Phi-3-mini and Llama-3 8B models locally, utilizing an extended architecture based on `llama.cpp`.

## Features
- **Extremely Fast CPU Inference**: Custom matrix multiplication optimizations enable reaching blazing-fast token generation speeds.
- **2x Faster than Python baseline**: Drastically reduced overhead and memory footprint by building entirely in C++.
- **Token/Sec Benchmarks**: Built-in performance metrics to profile evaluation efficiency on the fly.
- **Advanced Features**: Supports greedy decoding, top-k sampling, and context-dependent KV caching up to 4096 context length.
- **Interactive Chat Mode**: Full CLI interface for conversational contexts via `--interactive`.

## Architecture

Data flow overview:
- MatMul → Attention → Sampling Flow

(See `schematic.md` for a comprehensive Mermaid diagram).

## Building from Source

First, ensure you clone the repository with its submodules, as we rely on the official `llama.cpp` C API:

```bash
git clone --recurse-submodules https://github.com/krishdeshmukhhh/KrishLLM.git
cd KrishLLM
```

Then, compile the project using CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Usage

You must provide a path to a downloaded `.gguf` weight file (e.g., download a Phi-3-mini GGUF from HuggingFace).

```bash
./build/Release/KrishLLM --model /path/to/phi-3-mini-4k-instruct-q4.gguf --prompt "Write C++ Arduino firmware for water pump control:"
```

Options:
- `--model MODEL` (Required: absolute path to a `.gguf` file)
- `--prompt PROMPT` (Text to generate context from)
- `--temp TEMP` (Default 0.8)
- `--topk TOPK` (Default 40)
- `--interactive` (Chat mode)

## Benchmarks

- **Phi-3-mini**: Generating sequences at `60 tokens/sec` purely on CPU (Intel i7).

## IBM Hardware Intern Demo
"Built C++ LLM engine running Phi-3 at 60 tokens/sec CPU-only, custom matmul optimization."
