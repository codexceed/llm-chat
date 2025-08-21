# ADR-001: Initial Architecture Decision - Streamlit + OpenAI-compatible LLM API

## Status
Accepted

## Context
We needed to build a chatbot application that would serve as a learning project to understand state-of-the-art ML engineering concepts including inference optimization, LLMs, agentic systems, and distributed systems. The application required:

1. **User Interface**: An intuitive web interface for chat interactions
2. **LLM Integration**: Connection to large language models for conversational AI
3. **Rapid Development**: Quick iteration and prototyping capabilities
4. **Flexibility**: Support for multiple LLM providers and local models
5. **Learning Focus**: Technology choices that expose key ML engineering concepts

## Decision
We chose **Streamlit** as the web UI framework and **OpenAI-compatible API** as the LLM integration approach.

### UI Framework: Streamlit
- Streamlit provides rapid prototyping for data science and ML applications
- Native support for chat interfaces with `st.chat_message()` and `st.chat_input()`
- Built-in session state management for conversational context
- Minimal boilerplate code compared to Flask/FastAPI + frontend frameworks
- Excellent developer experience with hot reloading

### LLM Integration: OpenAI-compatible API
- Standard interface that works with multiple providers (OpenAI, Anthropic, local models)
- Streaming response support for better user experience
- Simple HTTP-based integration with well-established patterns
- Easy to swap between different models and providers
- Supports both cloud and self-hosted LLM deployments

## Alternatives Considered

### UI Framework Alternatives
1. **Flask/FastAPI + React/Vue**: More control but significantly more complexity
2. **Gradio**: Similar to Streamlit but less flexible for custom layouts
3. **Jupyter Widgets**: Good for experimentation but limited for production interfaces
4. **CLI-only**: Simple but poor user experience for chat interactions

### LLM Integration Alternatives
1. **Direct model loading (HuggingFace Transformers)**: More control but requires significant infrastructure
2. **LangChain abstractions**: Additional complexity layer over API calls
3. **Custom protocol**: Vendor lock-in and unnecessary complexity
4. **gRPC/WebSocket protocols**: More efficient but overkill for this use case

## Consequences

### Positive
- **Rapid Development**: Streamlit enables quick iteration on UI components
- **Learning Focus**: More time spent on ML concepts rather than web development
- **Flexibility**: Easy to experiment with different LLM providers
- **Streaming Support**: Real-time response rendering improves user experience
- **Session Management**: Built-in state handling for conversational context
- **File Upload**: Native support for document processing in RAG workflows

### Negative
- **UI Limitations**: Less control over styling and layout compared to custom frontends
- **Scaling Constraints**: Streamlit may not be optimal for high-traffic production use
- **Mobile Experience**: Limited responsive design capabilities
- **API Dependencies**: Requires network connectivity for LLM inference
- **Cost Considerations**: Cloud API usage can be expensive at scale

### Risks and Mitigations
- **Vendor Lock-in**: Mitigated by using OpenAI-compatible standard interface
- **Performance Bottlenecks**: Can migrate to custom frontend if needed
- **API Rate Limits**: Local model support provides fallback option
- **UI Complexity**: Streamlit limitations may require refactoring for advanced features

## Implementation Notes
- OpenAI client library provides consistent interface across providers
- Streaming responses implemented using `st.write_stream()` for real-time updates
- Session state manages conversation history and uploaded documents
- Environment-based configuration supports easy provider switching
- Docker Compose setup enables local LLM testing with vLLM

## Related Decisions
- [ADR-002: RAG System Architecture](002-rag-system-architecture.md) - Document processing integration
- [ADR-005: Configuration Management](005-configuration-management.md) - Environment-based provider configuration

## References
- [Streamlit Chat Elements Documentation](https://docs.streamlit.io/library/api-reference/chat)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)