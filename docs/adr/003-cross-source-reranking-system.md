# ADR-003: Cross-Source Re-ranking System

**Date:** 2025-08-17  
**Status:** Proposed  
**Deciders:** Engineering Team  
**Tags:** rag, web-search, ranking, performance

## Context

The current web search RAG integration combines results from internal knowledge base and web search through simple concatenation. This approach has several limitations:

- No intelligent prioritization of more relevant content
- Basic ordering (web first, then internal) regardless of actual relevance
- Missing authority and freshness considerations
- Potential for redundant or contradictory information in context
- Suboptimal use of limited context window space

As we scale the system and add more data sources, we need a sophisticated re-ranking mechanism to ensure the most relevant, authoritative, and fresh content reaches the LLM.

## Decision

We will implement a **Cross-Source Re-ranking System** that intelligently orders content from multiple sources (internal RAG + web search) using a multi-stage pipeline:

### Architecture Components

1. **Cross-encoder Re-ranking Model**
   - Primary re-ranking using semantic relevance scores
   - Model: Sentence-BERT or similar cross-encoder
   - Input: [query, document] pairs from all sources
   - Output: Unified relevance scores (0-1)

2. **Authority Weighting System**
   - Domain authority scoring (.edu, .gov, known publishers)
   - Internal document trust scores
   - Citation/link analysis for web content
   - Configurable authority multipliers (0.5x - 2.0x)

3. **Freshness Scoring Engine**
   - Time-decay functions for age-sensitive content
   - Smart temporal query detection
   - Configurable decay rates by content type
   - Fresh content boost for recent queries

4. **Final Score Computation**
   ```
   final_score = base_relevance_score × authority_weight × freshness_factor
   ```

### Implementation Strategy

#### Phase 1: Cross-encoder Integration (2-3 weeks)
- Integrate sentence-transformers/cross-encoder model
- Implement unified scoring for all content sources
- Basic re-ranking pipeline in `chatbot/ranking/`

#### Phase 2: Authority & Freshness (2 weeks)  
- Domain authority scoring system
- Content freshness analysis and scoring
- Configurable weighting parameters

#### Phase 3: Optimization (1-2 weeks)
- Performance optimization (batching, caching)
- A/B testing framework
- Monitoring and metrics collection

### Technical Decisions

**Cross-encoder Model Choice:**
- **Selected:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Rationale:** Good balance of accuracy and speed, optimized for search tasks
- **Alternative:** Fine-tune on domain-specific data later

**Skip RRF (Reciprocal Rank Fusion):**
- Cross-encoder provides unified scoring, making RRF redundant
- Simpler pipeline with better semantic understanding
- Direct authority/freshness integration

**Authority Sources:**
- Domain reputation databases (Alexa, Majestic)
- Manual curated lists for high-trust domains
- Internal document classification system

## Consequences

### Positive
- **Improved relevance:** Better context quality through semantic re-ranking
- **Source diversity:** Intelligent mixing of internal and external content  
- **Authority awareness:** Trusted sources get appropriate weighting
- **Freshness handling:** Time-sensitive queries get recent content
- **Scalable:** Framework supports additional ranking signals

### Negative
- **Increased latency:** Cross-encoder inference adds ~100-300ms
- **Computational cost:** GPU/CPU requirements for model inference
- **Complexity:** More sophisticated system to maintain and debug
- **Dependency:** Requires external model and potential network calls

### Risks & Mitigations

**Performance Impact:**
- *Risk:* Re-ranking latency affects user experience
- *Mitigation:* Batch processing, model optimization, caching

**Model Bias:**
- *Risk:* Cross-encoder may have domain bias
- *Mitigation:* Evaluation framework, potential fine-tuning

**Authority Gaming:**
- *Risk:* Authority scores could be manipulated
- *Mitigation:* Multiple authority signals, regular audits

## Implementation Details

### New Files/Modules
```
chatbot/ranking/
├── __init__.py
├── cross_encoder.py      # Cross-encoder model integration
├── authority.py          # Domain authority scoring
├── freshness.py          # Content freshness analysis
└── pipeline.py           # Main re-ranking orchestration

chatbot/evaluation/
├── __init__.py
└── ranking_metrics.py    # Evaluation metrics for ranking quality
```

### Configuration Changes
```python
# Add to chatbot/settings.py
class RankingSettings(BaseSettings):
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    authority_weight: float = 0.3
    freshness_weight: float = 0.2
    batch_size: int = 32
    enable_caching: bool = True
```

### Integration Points
- Modify `WebContextPipeline.merge_context()` to use re-ranking
- Update `chatbot/rag.py` to support re-ranking interface
- Extend settings configuration for ranking parameters

## Success Metrics

### Quality Metrics
- **Relevance improvement:** +20% in semantic similarity scores
- **User satisfaction:** A/B test with user feedback
- **Context utilization:** Better use of context window space

### Performance Metrics  
- **Latency target:** <200ms additional latency for re-ranking
- **Throughput:** Handle current query volume without degradation
- **Resource usage:** Monitor GPU/CPU utilization

### Evaluation Framework
- Automated relevance scoring using LLM-as-judge
- Human evaluation on sample queries
- Continuous monitoring of ranking quality metrics

## Alternatives Considered

1. **RRF + Simple Weighting**
   - *Rejected:* Less sophisticated than cross-encoder approach
   - *Reason:* Cross-encoder provides better semantic understanding

2. **Learning-to-Rank (LTR) Models**
   - *Rejected:* Requires extensive training data and feature engineering  
   - *Reason:* Cross-encoder is more practical for current scale

3. **Multiple Specialized Rankers**
   - *Rejected:* Increased complexity without clear benefits
   - *Reason:* Single cross-encoder with weighting is simpler and effective

## Related Documents
- [Web Search RAG Integration Proposal](../proposals/web_search_rag_integration.md)
- [ADR-002: Web Context Independence](002-web-context-independence.md)

## References
- [Sentence-BERT Cross-encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [MS MARCO Ranking Models](https://github.com/microsoft/MSMARCO-Passage-Ranking)
- [Information Retrieval Re-ranking Survey](https://arxiv.org/abs/2006.05324)