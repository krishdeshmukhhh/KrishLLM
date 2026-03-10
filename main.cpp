#include "llama.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

struct LLMArgs {
    std::string model = "";
    std::string prompt = "";
    float temp = 0.8f;
    int top_k = 40;
    bool interactive = false;
    int ctx_length = 4096;
};

void print_help() {
    std::cout << "KrishLLM - Real C++ Inference Engine (powered by llama.cpp)\n";
    std::cout << "Usage: ./KrishLLM [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --help          Print this help message\n";
    std::cout << "  --model MODEL   Path to the GGUF model file (REQUIRED)\n";
    std::cout << "  --prompt PROMPT Input prompt to start generation\n";
    std::cout << "  --temp TEMP     Temperature for sampling (default: 0.8)\n";
    std::cout << "  --topk TOPK     Top-k sampling (default: 40)\n";
}

bool parse_args(int argc, char** argv, LLMArgs& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return false;
        } else if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--temp" && i + 1 < argc) {
            args.temp = std::stof(argv[++i]);
        } else if (arg == "--topk" && i + 1 < argc) {
            args.top_k = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help();
            return false;
        }
    }
    
    if (args.model.empty()) {
        std::cerr << "Error: --model path is required.\n";
        print_help();
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    LLMArgs args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    // Initialize backend
    llama_backend_init();

    // Setup model parameters
    llama_model_params model_params = llama_model_default_params();
    std::cout << "[INFO] Loading model '" << args.model << "' into memory...\n";
    
    llama_model* model = llama_load_model_from_file(args.model.c_str(), model_params);
    if (model == NULL) {
        std::cerr << "Error: unable to load model\n";
        return 1;
    }

    // Setup context parameters (KV Cache size)
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = args.ctx_length;

    std::cout << "[INFO] Allocating KV cache (Context length: " << args.ctx_length << ")...\n";
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        std::cerr << "Error: failed to create the llama_context\n";
        llama_free_model(model);
        return 1;
    }

    std::cout << "[INFO] Model loaded successfully.\n";
    std::cout << "====================================================\n\n";

    if (args.prompt.empty()) {
        std::cout << "No prompt provided. Exiting.\n";
        llama_free(ctx);
        llama_free_model(model);
        return 0;
    }

    std::cout << "Prompt: " << args.prompt << "\n\n";
    std::cout << "Generating...\n\n";

    // Tokenize prompt
    std::vector<llama_token> tokens_list;
    tokens_list.resize(args.prompt.size() + 2); // Initial estimate
    
    int n_tokens = llama_tokenize(model, args.prompt.c_str(), args.prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
    if (n_tokens < 0) {
        tokens_list.resize(-n_tokens);
        n_tokens = llama_tokenize(model, args.prompt.c_str(), args.prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
    }
    tokens_list.resize(n_tokens);

    // Initial eval
    llama_batch batch = llama_batch_init(512, 0, 1);
    
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Error: llama_decode failed\n";
        return 1;
    }

    // Generation parameters
    int n_cur = batch.n_tokens;
    int n_decode = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();

    // Simple generation loop
    while (n_cur <= args.ctx_length) {
        auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        int n_vocab = llama_n_vocab(model);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        
        // Sampling
        llama_sample_top_k(ctx, &candidates_p, args.top_k, 1);
        llama_sample_temp(ctx, &candidates_p, args.temp);
        
        llama_token new_token_id = llama_sample_token(ctx, &candidates_p);

        // Break if End Of String
        if (llama_token_is_eog(model, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::cerr << "Error decoding token\n";
            break;
        }

        std::cout << std::string(buf, n) << std::flush;

        // Prepare next batch
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

        n_decode++;
        n_cur++;

        // Evaluate
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Error during evaluation\n";
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "\n\n====================================================\n";
    std::cout << "\n[BENCHMARK] Inference Complete.\n";
    std::cout << "[BENCHMARK] Tokens generated: " << n_decode << "\n";
    std::cout << "[BENCHMARK] Processing time: " << duration.count() << " seconds\n";
    std::cout << "[BENCHMARK] Speed: " << (n_decode / duration.count()) << " t/s\n";

    // Clean up
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
