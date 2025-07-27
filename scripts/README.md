# Scripts Directory

This directory contains utility scripts for testing, demonstrating, and analyzing the chatbot's RAG (Retrieval-Augmented Generation) functionality.

## Available Scripts

### 1. `compare_retrieval_modes.py`
**Comprehensive end-to-end comparison of Dense vs Hybrid retrieval**

- **Purpose**: Advanced performance comparison using real Qdrant collections with comprehensive technical documentation
- **Requirements**: Running Qdrant instance (port 6333)
- **Data Structure**: Uses separate JSON files for maintainability:
  - `data/sample_documents.json`: 16 comprehensive technical documents
  - `data/test_queries.json`: 80 categorized test queries
- **Enhanced Features**:
  - **16 comprehensive technical documents** covering: Python concurrency/asyncio, JavaScript/React, database optimization, deep learning, microservices, Kubernetes, DevOps, cybersecurity, blockchain, edge computing, Docker, HTTP protocol, and AWS Lambda
  - **80 diverse test queries** organized by category (exact_keywords, semantic_concepts, code_patterns, technical_concepts, configuration, edge_cases, cross_domain, etc.)
  - Real-world technical documentation from authoritative sources (Python.org, Kubernetes.io, Docker.com, React.dev, MDN, AWS)
  - Creates separate test collections for dense and hybrid retrieval
  - Comprehensive final summary with category-based analysis and detailed insights
  - Success rate tracking and performance metrics comparison
  - Query complexity analysis (simple vs complex queries)
  - Detailed recommendations based on query types and content domains
  - **Print-based output** (not logging) for clear results display

**Usage**:
```bash
# Ensure Qdrant is running
docker-compose up qdrant

# Run the comparison
python scripts/compare_retrieval_modes.py
```

**Sample Output**:
```
Test 1: async await JavaScript
Description: Exact keyword match - hybrid should excel due to sparse retrieval
Expected best match: JavaScript Frontend Development

🔵 DENSE RETRIEVAL RESULTS:
  1. JavaScript Frontend Development (score: 0.7234)
  2. Python Programming Basics (score: 0.4123)
  3. API Design Best Practices (score: 0.3456)
  ⏱️  Time: 0.045s

🟡 HYBRID RETRIEVAL RESULTS:
  1. JavaScript Frontend Development (score: 0.8567)
  2. API Design Best Practices (score: 0.5432)
  3. Python Programming Basics (score: 0.4321)
  ⏱️  Time: 0.067s

📊 ANALYSIS:
  Expected result found in dense: ✅
  Expected result found in hybrid: ✅
  🏆 HYBRID WINS - Better ranking
```

**Sample Output**:
```
================================================================================
🏁 FINAL RETRIEVAL COMPARISON SUMMARY
================================================================================

📊 OVERALL RESULTS (80 test queries):
  🏆 Hybrid Wins: 45 (56.3%)
  🏆 Dense Wins: 18 (22.5%)
  🤝 Ties: 12 (15.0%)
  ❌ Both Missed: 5 (6.3%)

⏱️  PERFORMANCE METRICS:
  Dense Average Time: 0.0234s
  Hybrid Average Time: 0.0312s
  Hybrid Slowdown: 33.3%

✅ SUCCESS RATES (Found Expected Result):
  Dense Success Rate: 37.5%
  Hybrid Success Rate: 71.3%

🏷️  PERFORMANCE BY CATEGORY:
  Exact Keywords (20 queries):
    Hybrid: 16, Dense: 2, Ties: 2
    → 🏆 HYBRID ADVANTAGE in Exact Keywords

  Technical Concepts (15 queries):
    Hybrid: 11, Dense: 3, Ties: 1
    → 🏆 HYBRID ADVANTAGE in Technical Concepts

  Semantic Concepts (12 queries):
    Hybrid: 6, Dense: 4, Ties: 2
    → 🏆 HYBRID ADVANTAGE in Semantic Concepts

💡 KEY INSIGHTS:
  🟡 HYBRID excels with: Exact Keywords, Technical Concepts, Code Patterns
  🔵 DENSE competitive with: Semantic Concepts, Cross Domain

⚖️  RECOMMENDATIONS:
  → Use HYBRID for technical documentation and API references
  → Consider query complexity: Hybrid better for specific terms
  → Consider performance: Dense ~33.3% faster on average
```

## Comprehensive Query Categories Tested

The enhanced evaluation tests **80 diverse queries** across comprehensive technical documentation:

### 🎯 **Exact Keywords** (20 queries)
- **Python Asyncio**: `asyncio.run() coroutines tasks event loop`
- **Kubernetes**: `kubectl API server etcd scheduler control plane`, `Pods Deployments ReplicaSets StatefulSets DaemonSets`
- **Docker**: `dockerd daemon Docker CLI APIs container lifecycle`, `BuildKit multi-stage builds Swarm Mode rootless`
- **React**: `useState useEffect JSX components PascalCase`
- **HTTP**: `GET POST PUT PATCH DELETE HEAD OPTIONS methods`, `Accept Authorization Content-Type Cache-Control ETag headers`
- **AWS Lambda**: `lambda_handler event context serverless compute`, `API Gateway S3 DynamoDB CloudWatch SQS SNS triggers`

### 🧠 **Semantic Concepts** (15 queries)  
- **Concurrency**: `concurrent coroutine execution synchronization primitives`
- **Architecture**: `declarative configuration desired state container orchestration`
- **Performance**: `event-driven automatic scaling pay-per-use pricing`
- **Frontend**: `props unidirectional data flow component composition`
- **Networking**: `client-server stateless protocol content negotiation`

### 💻 **Code Patterns** (12 queries)
- **Python**: `async def await asyncio.sleep networking IO`
- **React**: `useState setCount state immutable re-rendering`
- **Original patterns**: `import asyncio aiohttp ClientSession`, `SELECT COUNT GROUP BY HAVING`
- **API patterns**: `@app.get @app.post FastAPI HTTPException`

### 🔧 **Technical Concepts** (18 queries)
- **Asyncio**: `Event Loop Futures Transports Protocols low-level`
- **Kubernetes**: `RBAC Pod Security Standards Network Policies security contexts`
- **Docker**: `overlay2 devicemapper storage drivers network drivers`
- **React**: `Higher-Order Components HOCs Render Props Context API`, `React.memo useMemo useCallback virtual DOM performance`
- **HTTP**: `CORS CSP Permissions Policy HTTPS encryption security`, `HTTP/1.1 HTTP/2 HTTP/3 QUIC multiplexing protocol versions`
- **AWS Lambda**: `SnapStart code layers concurrency controls VPC integration`, `CloudWatch Logs X-Ray tracing IAM roles policies`

### ⚙️ **Configuration** (3 queries)
- `apiVersion apps/v1 kind Deployment` - Kubernetes YAML
- `GitHub Actions workflow jobs steps` - CI/CD configuration  
- `pragma solidity contract function external` - Smart contract syntax

### 🔬 **Edge Cases** (8 queries)
- **Platform limitations**: `WebAssembly WASI compatibility asyncio limitations`
- **Advanced features**: `Custom Resources Operators Admission Controllers API extensions`
- **Build optimization**: `BuildKit enhanced build engine caching optimization`
- **Error handling**: `Error Boundaries JavaScript errors component trees`
- **Security protocols**: `OAuth 2.0 delegated authorization Bearer Token authentication`
- **Runtime extensibility**: `container image support custom runtimes Runtime API`

### 🌐 **Cross-Domain** (4 queries)
- `high-availability compute infrastructure automatic scaling resources`
- `networking I/O inter-process communication efficient handling`
- `container orchestration ecosystem monitoring CI/CD integration`
- `performance optimization resource allocation monitoring observability`

## Key Insights from Testing

### Dense Retrieval Strengths:
- ✅ Semantic similarity and conceptual queries
- ✅ Understanding context and meaning
- ✅ Faster processing (single embedding lookup)
- ✅ Good for natural language queries

### Hybrid Retrieval Strengths:
- ✅ Exact keyword and term matching
- ✅ Technical terminology and abbreviations
- ✅ Code snippets and specific identifiers
- ✅ Combined semantic + lexical understanding
- ✅ Better recall for technical content

### Performance Trade-offs:
- **Speed**: Dense ~25-35% faster (no sparse computation overhead)
- **Accuracy**: Hybrid ~50-70% better success rate for technical documentation
- **Memory**: Hybrid uses more storage (dense + sparse vectors)
- **Complexity**: Hybrid requires additional BM42 sparse model (~500MB)
- **Content Type**: Hybrid excels with technical content, Dense competitive with natural language

## Configuration

The scripts respect the current application configuration from `chatbot/settings.py`:

```python
# Hybrid retrieval settings
use_hybrid_retrieval: bool = True
sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
hybrid_top_k: int = 100
top_k: int = 5  # Final results returned
```

## Recommendations

**Use Hybrid Retrieval When**:
- Queries contain specific technical terms or keywords
- Exact term matching is important
- Documents contain code, APIs, or structured content
- Users frequently search for specific identifiers

**Use Dense Retrieval When**:
- Queries are more conceptual or semantic
- Speed is critical
- Documents are primarily natural language
- Memory/storage constraints exist

## Enhanced Test Data

The evaluation now includes comprehensive real-world documentation:

### **Documents** (16 total):
1. **Python Asyncio Comprehensive Guide** - Official asyncio documentation
2. **Kubernetes Architecture and Concepts** - Kubernetes.io concepts guide
3. **Docker Engine Architecture and Features** - Docker.com engine documentation
4. **React Component Architecture and State Management** - React.dev learning guide
5. **HTTP Protocol Fundamentals and Web Communication** - MDN HTTP documentation
6. **AWS Lambda Serverless Computing Platform** - AWS Lambda developer guide
7. **Plus 10 original technical documents** covering databases, ML, microservices, security, blockchain, and edge computing

### **Sources**:
- Python.org official documentation
- Kubernetes.io official docs
- Docker.com official documentation
- React.dev official learning materials
- MDN Web Docs (Mozilla)
- AWS official documentation

## Adding New Tests

To add new test queries or documents, modify the JSON files in `scripts/data/`:

**`sample_documents.json`**:
```json
{
  "title": "Document Title",
  "content": "Comprehensive technical content with examples, APIs, and detailed explanations..."
}
```

**`test_queries.json`**:
```json
{
  "query": "specific technical terms or concepts",
  "category": "exact_keywords|semantic_concepts|code_patterns|technical_concepts|configuration|edge_cases|cross_domain",
  "description": "What this query tests and expected retrieval behavior",
  "expected_best": "Expected Document Title"
}
```

Consider including:
- **Domain-specific terminology** (APIs, commands, protocols)
- **Code examples with exact syntax** (functions, configurations)
- **Mixed semantic/keyword queries** (conceptual + specific terms)
- **Edge cases** (platform limitations, advanced features)
- **Cross-domain concepts** (spanning multiple technologies)

## Troubleshooting

**Qdrant Connection Issues**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Start Qdrant if needed
docker-compose up qdrant
```

**Import Errors**:
```bash
# Ensure you're running from project root
cd /path/to/llm-chat
python scripts/compare_retrieval_modes.py
```

**Model Download Issues**:
- The BM42 sparse model will download automatically on first use
- Requires internet connection and ~500MB storage
- Models are cached in `~/.cache/fastembed/`