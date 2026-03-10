```mermaid
graph TD
    A[Input Prompt] --> B[Tokenizer]
    B --> C[Token Embeddings]
    
    subgraph Transformer Block
    C --> D[Layer Normalization]
    D --> E[Multi-Head Attention]
    E --> F[Add & Norm]
    F --> G[Feed Forward Network]
    G --> H[Add & Norm]
    H --> I[Next Block/Layer]
    end
    
    subgraph KrishLLM Optimizations
    E ---|Custom MatMul| J[CPU SIMD Accelerated Matrix Operations]
    E ---|KV Cache (4096 ctx)| K[Efficient Memory Management]
    end
    
    I --> L[Logits]
    
    subgraph Sampling
    L --> M{Decoding Strategy}
    M -->|Top-K Sampling| N[Filter Top K Probs]
    M -->|Greedy Decoding| O[Select Max Prob]
    N --> P[Apply Temperature (0.8)]
    P --> Q[Sample Token]
    O --> Q
    end
    
    Q --> R[Output Token]
    R -->|Append to Context| A
```
