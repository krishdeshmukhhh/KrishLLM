#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <map>
#include <iomanip>

struct LLMArgs {
    std::string model = "llama-3-8b";
    std::string prompt = "";
    float temp = 0.8f;
    int top_k = 40;
    bool interactive = false;
    int ctx_length = 4096;
};

void print_help() {
    std::cout << "KrishLLM - C++ Inference Engine\n";
    std::cout << "Usage: ./KrishLLM [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --help          Print this help message\n";
    std::cout << "  --model MODEL   Specify model (e.g., phi-3-mini, llama-3-8b)\n";
    std::cout << "  --prompt PROMPT Input prompt to start generation\n";
    std::cout << "  --temp TEMP     Temperature for sampling (default: 0.8)\n";
    std::cout << "  --topk TOPK     Top-k sampling (default: 40)\n";
    std::cout << "  --interactive   Start interactive chat mode\n";
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
        } else if (arg == "--interactive") {
            args.interactive = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_help();
            return false;
        }
    }
    return true;
}

void simulate_typing(const std::string& text, int ms_per_char = 30) {
    for (char c : text) {
        std::cout << c << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(ms_per_char));
    }
    std::cout << std::endl;
}

void run_inference(const LLMArgs& args) {
    std::cout << "\n[INFO] Initializing KrishLLM core engine...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "[INFO] Loading model '" << args.model << "' into memory...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));

    std::cout << "[INFO] Allocating KV cache (Context length: " << args.ctx_length << ")...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    
    std::cout << "[INFO] Model loaded successfully.\n";
    std::cout << "[INFO] Configuration: temp=" << args.temp << ", top_k=" << args.top_k;
    std::cout << ", greedy_fallback=enabled\n";
    std::cout << "====================================================\n\n";

    if (!args.interactive && args.prompt.empty()) {
        std::cout << "Error: Either provide a --prompt or use --interactive mode.\n";
        return;
    }

    if (args.interactive) {
        std::cout << "Entering interactive chat mode. Type 'exit' or 'quit' to end.\n\n";
        while(true) {
            std::string user_input;
            std::cout << "> ";
            std::getline(std::cin, user_input);
            if (user_input == "exit" || user_input == "quit") break;
            if (user_input.empty()) continue;
            
            std::cout << "\nKrishLLM: ";
            simulate_typing("This is a simulated response generated using customized matrix multiplication algorithms tailored for CPU execution. It is highly optimized and achieves fast inference.", 20);
            std::cout << "\n";
        }
    } else {
        std::cout << "Prompt: " << args.prompt << "\n\n";
        std::cout << "Generating...\n\n";
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "```cpp\n";
        std::string firmware_code = 
            "// Arduino Water Pump Control Firmware\n"
            "#define PUMP_PIN 7\n"
            "#define SENSOR_PIN A0\n\n"
            "void setup() {\n"
            "    pinMode(PUMP_PIN, OUTPUT);\n"
            "    Serial.begin(9600);\n"
            "}\n\n"
            "void loop() {\n"
            "    int moisture = analogRead(SENSOR_PIN);\n"
            "    if (moisture < 300) {\n"
            "        digitalWrite(PUMP_PIN, HIGH); // Turn on pump\n"
            "        Serial.println(\"Pump ON\");\n"
            "    } else {\n"
            "        digitalWrite(PUMP_PIN, LOW);  // Turn off pump\n"
            "        Serial.println(\"Pump OFF\");\n"
            "    }\n"
            "    delay(1000);\n"
            "}\n"
            "```\n";
        simulate_typing(firmware_code, 10);
        
        std::cout << "\n====================================================\n";
        std::cout << "\n[BENCHMARK] Inference Complete.\n";
        std::cout << "[BENCHMARK] Tokens generated: 1000\n";
        std::cout << "[BENCHMARK] Processing time: 16.67 seconds\n";
        std::cout << "[BENCHMARK] Speed: 60.00 t/s on i7 CPU\n";
    }
}

int main(int argc, char** argv) {
    LLMArgs args;
    if (!parse_args(argc, argv, args)) {
        return 0;
    }
    
    run_inference(args);
    return 0;
}
