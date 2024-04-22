# prepare wiki embedding
- raglab need preprocess wiki2023 into colbert embedding.
## download dataset
- source data get from factscore repo(https://github.com/shmsw25/FActScore)
  - https://drive.google.com/file/d/1mekls6OGOKLmt7gYtHs0WGf5oTamTNat/view
## 



# dir tree
- cirrent raglab dir struction
'''
.
├── 1-eval_output
│   ├── ASQA
│   ├── Factscore
│   ├── PopQA
│   └── PubHealth
├── ALCE
│   ├── assets
│   ├── configs
│   ├── human_eval
│   ├── paper
│   ├── prompts
│   ├── run
│   └── tools
├── config
│   ├── active_rag
│   ├── iterative_rag
│   ├── naive_rag
│   ├── query_rewrite_rag
│   ├── self_ask
│   ├── selfrag_original
│   └── selfrag_reproduction
├── config-debug
│   ├── active_rag
│   ├── iterative_rag
│   ├── naive_rag
│   ├── query_rewrite_rag
│   ├── self_ask
│   ├── selfrag_original
│   └── selfrag_reproduction
├── data
│   ├── eval_datasets
│   │   ├── 2WikiMultiHopQA
│   │   ├── ASQA
│   │   ├── Arc_Challenge
│   │   ├── Factscore
│   │   ├── HotPotQA
│   │   ├── MMLU
│   │   │   └── data
│   │   │       ├── auxiliary_train
│   │   │       ├── dev
│   │   │       ├── test
│   │   │       └── val
│   │   ├── PopQA
│   │   ├── PubHealth
│   │   ├── StrategyQA
│   │   ├── TriviaQA
│   │   └── feverous
│   ├── eval_results
│   ├── retrieval
│   │   ├── colbertv2.0_embedding
│   │   │   ├── indexes
│   │   │   │   ├── colbertv2.0_embedding
│   │   │   │   └── lifestyle.dev.2bits
│   │   │   ├── wiki2023
│   │   │   │   └── indexes
│   │   │   │       └── wiki2023
│   │   │   └── wikipedia2018
│   │   │       └── indexes
│   │   │           └── wikipedia2018
│   │   ├── colbertv2.0_passages
│   │   │   ├── lotte
│   │   │   │   ├── lifestyle
│   │   │   │   │   ├── dev
│   │   │   │   │   └── test
│   │   │   │   ├── pooled
│   │   │   │   │   ├── dev
│   │   │   │   │   └── test
│   │   │   │   ├── recreation
│   │   │   │   │   ├── dev
│   │   │   │   │   └── test
│   │   │   │   ├── science
│   │   │   │   │   ├── dev
│   │   │   │   │   └── test
│   │   │   │   ├── technology
│   │   │   │   │   ├── dev
│   │   │   │   │   └── test
│   │   │   │   └── writing
│   │   │   │       ├── dev
│   │   │   │       └── test
│   │   │   ├── wiki2023
│   │   │   └── wikipedia2018
│   │   ├── contriever_embedding
│   │   │   ├── wikipedia_embeddings
│   │   │   └── wikipedia_embeddings-debug
│   │   └── contriever_passages
│   └── train_data
├── evaluation
├── local_cache
│   └── compiler
├── model
│   ├── Llama-2-7b-chat-hf
│   ├── Llama-2-7b-hf
│   ├── PandaLM-Alpaca-7B-v1
│   ├── colbertv2.0
│   ├── contriever-msmarco
│   ├── llama-7b-hf
│   └── selfrag_llama2_7b
├── postprocess
├── preprocess
│   ├── colbert-wiki2018-preprocess
│   ├── colbert-wiki2023-preprocess
│   └── datasets-preprocess
├── raglab
│   ├── dataset
│   │   └── base_dataset
│   ├── instruction_lab
│   ├── language_model
│   ├── rag
│   │   └── infer_alg
│   │       ├── active_rag
│   │       ├── iterative_rag
│   │       ├── naive_rag
│   │       ├── query_rewrite_rag
│   │       ├── self_ask
│   │       ├── self_rag_original
│   │       └── self_rag_reproduction
│   └── retrieval
│       ├── colbert
│       └── contriever
│           └── src
├── run
├── run-debug
└── test
'''