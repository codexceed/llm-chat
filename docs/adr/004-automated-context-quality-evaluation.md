# ADR-004: Automated Context Quality Evaluation

**Date:** 2025-08-17  
**Status:** Proposed  
**Deciders:** Engineering Team  
**Tags:** evaluation, quality-assurance, monitoring, llm-judge

## Context

As the RAG system evolves with web search integration and cross-source re-ranking, we need systematic ways to measure and maintain context quality. Current challenges include:

- No objective metrics for context relevance and quality
- Manual evaluation doesn't scale with query volume
- Difficulty detecting quality regressions after system changes
- No continuous monitoring of retrieval effectiveness
- Limited feedback for improving ranking algorithms

Without automated evaluation, we risk deploying changes that degrade user experience or fail to catch quality issues until users complain.

## Decision

We will implement an **Automated Context Quality Evaluation System** using a multi-metric approach with LLM-as-judge patterns and statistical monitoring.

### Architecture Components

1. **Multi-Metric Evaluation Framework**
   - **Relevance scoring:** Semantic similarity + LLM judgment  
   - **Information density:** Content-to-noise ratio analysis
   - **Factual consistency:** Contradiction detection within context
   - **Source authority:** Domain reputation and trust scoring
   - **Freshness assessment:** Content recency for temporal queries

2. **LLM Judge System**
   - **Primary judge:** Llama 3.1 8B (local, fast, cost-effective)
   - **Premium judge:** GPT-4o Mini (complex cases, ground truth)
   - **Specialized models:** PairRM for ranking comparison tasks
   - **Hybrid routing:** Automatic escalation for difficult evaluations

3. **Continuous Quality Monitoring**
   - Real-time quality score tracking
   - Quality drift detection algorithms  
   - Threshold-based alerting system
   - A/B testing framework for system improvements

4. **Evaluation Data Pipeline**
   - Query classification for evaluation strategy
   - Batch evaluation for historical analysis
   - Streaming evaluation for real-time monitoring
   - Ground truth dataset maintenance

### Implementation Strategy

#### Phase 1: Core Evaluation Framework (2-3 weeks)
- Multi-metric evaluation engine in `chatbot/evaluation/`
- Basic LLM judge integration with local models
- Relevance and consistency scoring pipelines

#### Phase 2: LLM Judge System (2 weeks)
- Hybrid judge routing (local + API)
- Specialized evaluation prompts and parsers
- Performance optimization and caching

#### Phase 3: Monitoring & Alerting (1-2 weeks)
- Real-time quality tracking dashboard
- Automated alerting for quality degradation
- A/B testing framework integration

### Technical Decisions

**Primary LLM Judge Selection:**
- **Selected:** Llama 3.1 8B Instruct (local deployment)
- **Rationale:** Best balance of accuracy, speed, and cost
- **Deployment:** vLLM or similar for optimized inference

**Premium Judge for Complex Cases:**
- **Selected:** GPT-4o Mini
- **Use cases:** Contradictory information, nuanced relevance, ground truth
- **Trigger:** Confidence threshold or manual escalation

**Evaluation Metrics:**
- **Primary:** Relevance score (0-10 scale)
- **Secondary:** Information density, authority, freshness, consistency
- **Composite:** Weighted overall quality score

**Quality Monitoring:**
- **Real-time:** Per-query quality scores with streaming aggregation
- **Alerting:** Statistical process control for drift detection
- **Reporting:** Daily/weekly quality trend analysis

## Consequences

### Positive
- **Objective quality measurement:** Data-driven context quality insights
- **Continuous improvement:** Automated feedback for ranking algorithms
- **Quality assurance:** Early detection of system regressions  
- **Scalable evaluation:** Handles high query volumes automatically
- **A/B testing enablement:** Quantitative comparison of system variants

### Negative
- **Infrastructure complexity:** Additional services and dependencies
- **Computational overhead:** LLM judge inference costs
- **Evaluation accuracy:** Potential biases in LLM judgments
- **Maintenance burden:** Keeping evaluation prompts and thresholds tuned

### Risks & Mitigations

**LLM Judge Reliability:**
- *Risk:* Inconsistent or biased evaluations from judge models
- *Mitigation:* Multi-judge consensus, human validation samples, prompt engineering

**Evaluation Latency:**
- *Risk:* Real-time evaluation impacts user experience
- *Mitigation:* Async evaluation pipeline, batch processing, model optimization

**Cost Management:**
- *Risk:* High API costs for premium judge usage
- *Mitigation:* Smart routing, local models primary, cost monitoring

## Implementation Details

### New Files/Modules
```
chatbot/evaluation/
├── __init__.py
├── metrics.py            # Core evaluation metrics
├── judges/
│   ├── __init__.py
│   ├── llm_judge.py      # LLM-based evaluation
│   ├── hybrid_judge.py   # Multi-model routing  
│   └── prompts.py        # Evaluation prompt templates
├── monitoring/
│   ├── __init__.py
│   ├── quality_tracker.py # Real-time quality monitoring
│   ├── drift_detector.py  # Statistical drift detection
│   └── alerting.py        # Quality degradation alerts
└── ab_testing.py         # A/B testing framework
```

### Configuration Changes
```python
# Add to chatbot/settings.py
class EvaluationSettings(BaseSettings):
    # LLM Judge Configuration
    primary_judge_model: str = "llama-3.1-8b-instruct"
    premium_judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.1
    premium_judge_threshold: float = 0.7
    
    # Quality Monitoring
    quality_threshold: float = 6.0
    drift_detection_window: int = 100
    alert_email_recipients: list[str] = []
    
    # Evaluation Metrics
    relevance_weight: float = 0.4
    consistency_weight: float = 0.2
    authority_weight: float = 0.2
    freshness_weight: float = 0.1
    density_weight: float = 0.1
```

### Evaluation Prompt Templates
```python
RELEVANCE_JUDGE_PROMPT = """
Rate how well this context answers the user's query on a scale of 1-10.

Query: {query}
Context: {context}

Evaluation criteria:
- Direct relevance to the question (40%)
- Completeness of information (30%) 
- Accuracy and reliability (30%)

Score (1-10): [your score]
Reasoning: [brief explanation]
"""

CONSISTENCY_JUDGE_PROMPT = """
Analyze this context for internal contradictions or conflicting information.

Context: {context}

Rate consistency on a scale of 1-10:
- 10: No contradictions, all information aligns
- 5: Minor inconsistencies that don't affect main points
- 1: Major contradictions that undermine reliability

Score (1-10): [your score]
Issues: [list any contradictions found]
"""
```

### Integration Points
- Extend `WebContextPipeline` to trigger evaluation after context assembly
- Integrate with `chatbot/rag.py` for internal document quality assessment
- Add evaluation hooks to re-ranking pipeline for continuous improvement
- Connect to monitoring dashboard for quality visualization

## Success Metrics

### Evaluation System Performance
- **Accuracy:** 85%+ agreement with human evaluators on sample set
- **Coverage:** 100% of queries evaluated within SLA
- **Latency:** <2s for standard evaluation, <5s for premium judge
- **Cost:** <$0.01 per evaluation average cost

### Quality Improvement Outcomes
- **Detection sensitivity:** Catch 90%+ of quality regressions within 24h
- **False positive rate:** <5% for quality degradation alerts
- **Correlation:** Strong correlation between automated scores and user satisfaction

### Operational Metrics
- **Availability:** 99.5%+ uptime for evaluation service
- **Throughput:** Handle 10x current query volume
- **Response time:** Sub-second quality score delivery

## Alternatives Considered

1. **Human-only Evaluation**
   - *Rejected:* Doesn't scale, expensive, slow feedback cycle
   - *Reason:* Need automated system for continuous monitoring

2. **Simple Metrics Only (No LLM Judge)**
   - *Rejected:* Misses semantic relevance and nuanced quality aspects
   - *Reason:* LLM judgment provides crucial semantic understanding

3. **Single High-end Model (GPT-4 only)**
   - *Rejected:* Too expensive for high-volume evaluation
   - *Reason:* Hybrid approach balances cost and accuracy

4. **Rule-based Evaluation**
   - *Rejected:* Limited ability to assess semantic relevance
   - *Reason:* Modern context quality requires semantic understanding

## Related Documents
- [ADR-003: Cross-Source Re-ranking System](003-cross-source-reranking-system.md)
- [Web Search RAG Integration Proposal](../proposals/web_search_rag_integration.md)

## References
- [LLM-as-Judge Best Practices](https://arxiv.org/abs/2306.05685)
- [RAGAS: Automated Evaluation of RAG Pipelines](https://github.com/explodinggradients/ragas)
- [Statistical Process Control for ML Systems](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)