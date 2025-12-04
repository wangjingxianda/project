This project implements a RAG-based (Retrieval-Augmented Generation) question–answering system for intelligent vehicle manuals. It uses existing car owner manuals to build a knowledge base, retrieves relevant information, and generates accurate answers with LLMs. The whole pipeline contains three major parts: knowledge base construction, information retrieval, and answer generation.

### 1、代码结构

``text`
.
├── Dockerfile                     # 镜像文件
├── README.md                      # 说明文档
├── bm25_retriever.py              # BM25召回
├── build.sh                       # 镜像编译打包
├── data                           # 数据目录
│   ├── result.json                # 结果提交文件
│   ├── test_question.json         # 测试集
│   └── train_a.pdf                # 训练集
├── faiss_retriever.py             # faiss向量召回
├── vllm_model.py                  # vllm大模型加速wrapper
├── pdf_parse.py                   # pdf文档解析器
├── pre_train_model                # 预训练大模型
│   ├── Qwen-7B-Chat               # Qwen-7B
│   │   └── download.py
│   ├── bge-reranker-large         # bge重排序模型
│   └── m3e-large                  # 向量检索模型
├── qwen_generation_utils.py       # qwen答案生成的工具函数
├── requirements.txt               # 此项目的第三方依赖库
├── rerank_model.py                # 重排序逻辑
├── run.py                         # 主文件                         
└── run.sh                         # 主运行脚本             
```

2. Project Overview
2.1 LLM-based Document Retrieval QA

The system answers car-related questions using information from vehicle manuals.
The pipeline finds relevant passages in the manual and asks an LLM to generate an answer.

Example:

Q1: How do I turn on the hazard warning lights?
A1: The switch is located below the steering wheel. Press it to activate the hazard lights.

Q2: How should I maintain the vehicle?
A2: Please follow regular maintenance steps, including cleaning, tire checks, and battery inspection.

Q3: The seat back feels too hot. What should I do?
A3: You can disable the seat heating function from the climate control menu.

3. Solution Design
3.1 PDF Parsing
3.1.1 Block-based PDF parsing

We parse the PDF into block-shaped segments to preserve content completeness.

3.1.2 Sliding-window parsing

Some important information spans across pages.
To keep context continuity, we treat the entire PDF as one long text string, split it by punctuation, and apply a sliding window.

Example:

Input array:

["aa","bb","cc","dd"]


Sliding window output:

aabb
bbcc
ccdd


The sliding window behaves like convolution with adjustable kernel size and stride. It maximizes coverage and avoids cutting through important steps.

3.1.3 Parsing strategy used in this project

We combine three parsing methods:

Block parsing using small headings + content (block sizes: 512 and 1024).

Sliding-window parsing on sentence segments (sizes: 256 and 512).

Non-sliding parsing with fixed-length splits (sizes: 256 and 512).

After parsing, we deduplicate the text blocks and send them to the retrieval module.

3.2 Retrieval Module

We use both vector retrieval (FAISS) and BM25 retrieval.

Vector retrieval captures semantic similarity.
BM25 captures keyword-level relevance.

They complement each other and are both widely used in industry.

3.2.1 Vector Retrieval

We use FAISS for indexing and M3E-large as the embedding model.

M3E (Moka Massive Mixed Embedding) characteristics:

trained on >22M Chinese pairs

supports Chinese–English mixed embeddings

strong semantic retrieval performance

open-source

We use the large version for best performance.

3.2.2 BM25 Retrieval

BM25 computes lexical similarity between the query and documents.
It is widely used for keyword-based search and provides strong recall for entity-heavy queries.

We use the BM25 retriever from LangChain.

3.3 Reranking

Reranking evaluates the relevance of retrieved documents and sorts them again.
This step improves retrieval accuracy before feeding context to the LLM.

Vector retrieval uses a bi-encoder, while reranking uses a cross-encoder, which is more accurate.

Pipeline:

Retrieve top-K documents (vector + BM25).

Use a reranker (cross-encoder) to score relevance.

Select the best passages as LLM input.

This improves LLM answer accuracy by reducing irrelevant context and lowering inference cost.

3.3.1 Cross-encoder Reranker

We use bge-reranker-large, an open-source high-performance reranker comparable to commercial models like Cohere Rerank.

Rerankers significantly improve precision, especially when combined with dense embeddings.

3.4 Inference Optimization — vLLM Acceleration

vLLM is a high-performance inference engine for large language models.

Its main advantages:

PagedAttention for efficient memory usage

Continuous batching

CUDA kernel optimizations

Multi-GPU and distributed inference

vLLM can nearly double inference speed compared with standard transformers-based inference.

We wrap Qwen-7B inference with vLLM through vllm_model.py.



3.5 run code

1. vscode run：python run.py
2. docker run：run bash build.sh first，then docker run  $name


























>>>>>>> master
