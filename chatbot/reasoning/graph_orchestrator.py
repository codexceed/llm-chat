"""LangGraph-based multi-step reasoning orchestrator."""

import datetime
import enum
from typing import Annotated, Any, Literal, TypedDict

import httpx
import openai
import pydantic
import scipy.spatial.distance
import streamlit as st
from langgraph import graph
from langgraph.graph import state as graph_state
from streamlit import logger
from streamlit.elements.lib import mutable_status_container

from chatbot import resources, settings
from chatbot.reasoning import templates
from chatbot.web import context

LOGGER = logger.get_logger("streamlit")


class StepType(enum.Enum):
    """Types of reasoning steps."""

    SEARCH = "search"
    REFINE = "refine"
    SYNTHESIZE = "synthesize"


class ReasoningStep(pydantic.BaseModel):
    """Represents a single reasoning step."""

    step_type: StepType
    query: str
    focus: str = ""
    rationale: str = ""


class ReasoningPlanSchema(pydantic.BaseModel):
    """Pydantic schema for the complete reasoning plan."""

    steps: list[ReasoningStep]


class StepResult(pydantic.BaseModel):
    """Result from executing a reasoning step."""

    step: ReasoningStep
    content: str
    success: bool
    error_message: str = ""
    summary: str | None = None
    query: str
    synthesized_from_indices: list[int] = pydantic.Field(default_factory=list)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def token_count(self) -> int:
        """Count tokens in content using cached tokenizer.

        Returns:
            Number of tokens in content
        """
        if not self.content:
            return 0
        try:
            tokenizer = resources.get_tokenizer()
            return len(tokenizer.encode(self.content, add_special_tokens=False))
        except (ValueError, TypeError, RuntimeError):
            # Fallback: rough estimate
            return len(self.content.split())

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def relevance_score(self) -> float:
        """Calculate relevance score to original query using embeddings.

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not self.query or not self.content:
            return 0.5

        try:
            # Generate embeddings on-demand using cached model
            embedding_model = resources.get_embedding_model()
            content_embedding = embedding_model.encode(self.content, convert_to_numpy=True)
            query_embedding = embedding_model.encode(self.query, convert_to_numpy=True)

            # Calculate cosine similarity
            similarity = 1 - scipy.spatial.distance.cosine(content_embedding, query_embedding)
            return float(similarity)
        except (ValueError, TypeError, RuntimeError):
            return 0.5


class ReasoningState(TypedDict):
    """State for the reasoning graph."""

    query: str
    planned_steps: list[ReasoningStep]
    current_step_index: int
    step_results: Annotated[list[StepResult], lambda x, y: x + y]
    final_response: str | None
    error_message: str | None


class GraphOrchestrator:
    """LangGraph-based orchestrator for multi-step reasoning chains."""

    def __init__(
        self,
        web_context_pipeline: context.WebContextPipeline,
        openai_client: openai.AsyncOpenAI,
        http_client: httpx.AsyncClient,
        model_name: str,
        seed: int,
        reasoning_settings: settings.MultiStepReasoningSettings,
    ) -> None:
        """Initialize the graph orchestrator.

        Args:
            web_context_pipeline: Pipeline for web context processing
            openai_client: OpenAI client for LLM interactions
            http_client: HTTP client for web requests
            model_name: LLM model name to use for reasoning
            seed: Random seed for reproducibility
            reasoning_settings: Multi-step reasoning settings configuration
        """
        self._web_pipeline = web_context_pipeline
        self._openai_client = openai_client
        self._http_client = http_client
        self._model = model_name
        self._settings = reasoning_settings
        self._seed = seed
        self._status_ui: mutable_status_container.StatusContainer | None = None

        # Build the graph
        self._graph = self._build_graph()

    def _build_graph(self) -> graph_state.CompiledStateGraph[ReasoningState, None, ReasoningState, ReasoningState]:
        """Build the reasoning graph.

        Returns:
            Compiled LangGraph StateGraph for reasoning workflow
        """
        state_graph = graph.StateGraph[ReasoningState, None, ReasoningState, ReasoningState](ReasoningState)

        # Add nodes
        state_graph.add_node("planner", self._planner_node)
        state_graph.add_node("search", self._search_node)
        state_graph.add_node("refine", self._refine_node)
        state_graph.add_node("synthesize", self._synthesize_node)
        state_graph.add_node("final_synthesis", self._final_synthesis_node)

        # Add edges
        state_graph.add_edge(graph.START, "planner")
        state_graph.add_conditional_edges("planner", self._after_planner, {"continue": "search", "end": graph.END})
        state_graph.add_conditional_edges(
            "search", self._step_router, {"search": "search", "refine": "refine", "synthesize": "synthesize", "end": "final_synthesis"}
        )
        state_graph.add_conditional_edges(
            "refine", self._step_router, {"search": "search", "refine": "refine", "synthesize": "synthesize", "end": "final_synthesis"}
        )
        state_graph.add_conditional_edges(
            "synthesize", self._step_router, {"search": "search", "refine": "refine", "synthesize": "synthesize", "end": "final_synthesis"}
        )
        state_graph.add_edge("final_synthesis", graph.END)

        return state_graph.compile()

    async def execute_complex_query(self, query: str, status_ui: mutable_status_container.StatusContainer | None = None) -> str | None:
        """Execute a complex query using the graph orchestrator.

        Args:
            query: The complex user query
            status_ui: Optional Streamlit status container for live updates

        Returns:
            Synthesized response from multi-step reasoning, if successful. Otherwise, None.
        """
        LOGGER.info("Starting graph-based multi-step reasoning for query: %s...", query[:100])

        # Store status_ui as instance variable for all nodes to access
        self._status_ui = status_ui

        initial_state = ReasoningState(
            query=query,
            planned_steps=[],
            current_step_index=0,
            step_results=[],
            final_response=None,
            error_message=None,
        )

        try:
            final_state = await self._graph.ainvoke(initial_state)
            return final_state.get("final_response")
        except (RuntimeError, ValueError, TypeError) as e:
            LOGGER.exception("Graph execution failed: %s", e)
            return None

    async def _planner_node(self, state: ReasoningState) -> dict[str, Any]:
        """Plan the reasoning steps for a complex query.

        Returns:
            Dictionary with planned_steps or error_message
        """
        LOGGER.info("Planning reasoning steps")

        # Update UI status
        if self._status_ui:
            self._status_ui.update(label="Planning reasoning steps…", state="running", expanded=False)

        timestamp = datetime.datetime.now().isoformat()
        planning_prompt = templates.PLANNING_PROMPT.format(
            timestamp=timestamp,
            max_steps=self._settings.max_steps,
            query=state["query"],
        )

        try:
            response = await self._openai_client.chat.completions.parse(
                model=self._model,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=self._settings.planning_temperature,
                seed=self._seed,
                response_format=ReasoningPlanSchema,
            )

            if response.choices[0].message.refusal:
                LOGGER.error("Planning LLM refused to generate response: %s", response.choices[0].message.refusal)
                return {"error_message": "Planning refused by LLM"}

            plan_content = response.choices[0].message.parsed
            if not plan_content:
                LOGGER.error("Planning LLM returned no parsed content")
                return {"error_message": "No plan generated"}

            LOGGER.info("Planned %d reasoning steps", len(plan_content.steps))
            LOGGER.debug("Planned steps: %s", plan_content.model_dump_json(indent=2))

            # Update UI with planned steps and show them to user
            if self._status_ui:
                self._status_ui.update(label=f"Planned {len(plan_content.steps)} steps. Executing…", state="running", expanded=True)

            # Display planned steps to user (similar to original orchestrator)
            st.write("Planned Steps:")
            for i, step in enumerate(plan_content.steps, 1):
                focus = f" — focus: {step.focus}" if step.focus else ""
                st.write(f"{i}. {step.step_type.value.upper()}: {step.query}{focus}")

            return {"planned_steps": plan_content.steps}

        except (RuntimeError, ValueError, TypeError) as e:
            LOGGER.exception("Planning failed: %s", e)

            # Update UI with error
            if self._status_ui:
                self._status_ui.update(label="Planning failed.", state="error", expanded=True)

            return {"error_message": f"Planning failed: {e}"}

    async def _search_node(self, state: ReasoningState) -> dict[str, Any]:
        """Execute a search step.

        Returns:
            Dictionary with step_results and updated current_step_index
        """
        current_step = state["planned_steps"][state["current_step_index"]]
        step_num = state["current_step_index"] + 1
        total_steps = len(state["planned_steps"])

        LOGGER.info("Executing search step: %s", current_step.query)

        # Update UI status
        if self._status_ui:
            self._status_ui.update(label=f"Executing step {step_num}/{total_steps}: {current_step.step_type.value}", expanded=True)

        st.write(f"\n▶ Step {step_num}: {current_step.step_type.value.upper()} — query: {current_step.query}")

        try:
            web_context_dict = await self._web_pipeline.gather_web_context(
                current_step.query,
                self._http_client,
                enable_web_search=True,
                force_web_search=True,
                search_num_results=self._settings.search_top_k,
            )

            search_content = web_context_dict.get("web_search", "")
            if not search_content:
                LOGGER.warning("No search results obtained")

            # Create step result
            result = StepResult(
                step=current_step,
                content=search_content,
                success=bool(search_content),
                query=current_step.query,
            )

            # Summarize if needed
            if result.success:
                result.summary = await self._summarize_content(result.content, current_step.query)

            # Update UI with step completion
            if self._status_ui:
                if result.success:
                    st.write(f"✅ Completed step {step_num}: {current_step.step_type.value}")
                else:
                    st.write(f"❌ Step {step_num} failed: {result.error_message}")

            return {
                "step_results": [result],
                "current_step_index": state["current_step_index"] + 1,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            LOGGER.exception("Search step failed: %s", e)
            result = StepResult(
                step=current_step,
                content="",
                success=False,
                error_message=str(e),
                query=current_step.query,
            )

            # Update UI with error
            if self._status_ui:
                st.write(f"❌ Step {step_num} error: {e}")

            return {
                "step_results": [result],
                "current_step_index": state["current_step_index"] + 1,
            }

    async def _refine_node(self, state: ReasoningState) -> dict[str, Any]:
        """Execute a refinement step.

        Returns:
            Dictionary with step_results and updated current_step_index

        Raises:
            ValueError: If refinement LLM returns empty response
        """
        current_step = state["planned_steps"][state["current_step_index"]]
        step_num = state["current_step_index"] + 1
        total_steps = len(state["planned_steps"])

        LOGGER.info("Executing refine step with focus: %s", current_step.focus)

        # Update UI status
        # status_ui now accessed via self._status_ui
        if self._status_ui:
            self._status_ui.update(label=f"Executing step {step_num}/{total_steps}: {current_step.step_type.value}", expanded=True)

        st.write(f"\n▶ Step {step_num}: {current_step.step_type.value.upper()} — query: {current_step.query}")

        try:
            # Get previous successful results
            successful_results = [r for r in state["step_results"] if r.success]

            if self._settings.enable_context_compression:
                previous_content = "\n\n".join([r.summary or "" for r in successful_results])
            else:
                previous_content = "\n\n".join([r.content for r in successful_results])

            refine_prompt = templates.REFINE_PROMPT.format(
                focus=current_step.focus,
                query=current_step.query,
                previous_content=previous_content,
            )

            # Get refined query from LLM
            response = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.3,
                max_tokens=200,
            )

            refined_query = response.choices[0].message.content or ""
            if not refined_query:
                LOGGER.debug("Empty response from refinement LLM: %s", response.model_dump_json(indent=2))
                raise ValueError("Empty response from refinement LLM")

            refined_query = refined_query.strip()
            LOGGER.debug("Refined query: %s", refined_query)

            # Execute search with refined query
            web_context_dict = await self._web_pipeline.gather_web_context(
                refined_query,
                self._http_client,
                enable_web_search=True,
                force_web_search=True,
                search_num_results=self._settings.search_top_k,
            )

            search_content = web_context_dict.get("web_search", "")

            result = StepResult(
                step=current_step,
                content=search_content,
                success=bool(search_content),
                query=current_step.query,
            )

            if result.success:
                result.summary = await self._summarize_content(result.content, current_step.query)

            # Update UI with step completion
            if self._status_ui:
                if result.success:
                    st.write(f"✅ Completed step {step_num}: {current_step.step_type.value}")
                else:
                    st.write(f"❌ Step {step_num} failed: {result.error_message}")

            return {
                "step_results": [result],
                "current_step_index": state["current_step_index"] + 1,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            LOGGER.exception("Refine step failed: %s", e)
            result = StepResult(
                step=current_step,
                content="",
                success=False,
                error_message=str(e),
                query=current_step.query,
            )

            # Update UI with error
            if self._status_ui:
                st.write(f"❌ Step {step_num} error: {e}")

            return {
                "step_results": [result],
                "current_step_index": state["current_step_index"] + 1,
            }

    async def _synthesize_node(self, state: ReasoningState) -> dict[str, Any]:
        """Execute a synthesis step.

        Returns:
            Dictionary with step_results and updated current_step_index

        Raises:
            ValueError: If no successful results are available to synthesize
        """
        current_step = state["planned_steps"][state["current_step_index"]]
        step_num = state["current_step_index"] + 1
        total_steps = len(state["planned_steps"])

        LOGGER.info("Executing synthesize step")

        # Update UI status
        # status_ui now accessed via self._status_ui
        if self._status_ui:
            self._status_ui.update(label=f"Executing step {step_num}/{total_steps}: {current_step.step_type.value}", expanded=True)

        st.write(f"\n▶ Step {step_num}: {current_step.step_type.value.upper()} — query: {current_step.query}")

        try:
            # Find which results have already been incorporated into synthesized steps
            already_synthesized_indices = set()
            for result in state["step_results"]:
                if result.step.step_type == StepType.SYNTHESIZE and result.success:
                    already_synthesized_indices.update(result.synthesized_from_indices)

            # Only include results that haven't been synthesized yet, plus any synthesized results
            combined_content = ""
            synthesized_indices = []
            for i, result in enumerate(state["step_results"]):
                if result.success and (result.step.step_type == StepType.SYNTHESIZE or i not in already_synthesized_indices):
                    combined_content += f"=== Step {i + 1} Results ===\n{result.content}\n\n"
                    synthesized_indices.append(i)

            if not combined_content:
                raise ValueError("No successful results to synthesize")

            result = StepResult(
                step=current_step,
                content=combined_content,
                success=True,
                query=current_step.query,
                synthesized_from_indices=synthesized_indices,
            )

            if result.success:
                result.summary = await self._summarize_content(result.content, current_step.query)

            # Update UI with step completion
            if self._status_ui:
                if result.success:
                    st.write(f"✅ Completed step {step_num}: {current_step.step_type.value} (SYNTH)")
                else:
                    st.write(f"❌ Step {step_num} failed: {result.error_message}")

            return {
                "step_results": [result],
                "current_step_index": state["current_step_index"] + 1,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            LOGGER.exception("Synthesize step failed: %s", e)
            result = StepResult(
                step=current_step,
                content="",
                success=False,
                error_message=str(e),
                query=current_step.query,
            )

            # Update UI with error
            if self._status_ui:
                st.write(f"❌ Step {step_num} error: {e}")

            return {
                "step_results": [result],
                "current_step_index": state["current_step_index"] + 1,
            }

    async def _final_synthesis_node(self, state: ReasoningState) -> dict[str, Any]:
        """Synthesize final response from all step results.

        Returns:
            Dictionary with final_response or error_message
        """
        LOGGER.info("Synthesizing final response from %d steps", len(state["step_results"]))

        # Update UI status
        # status_ui now accessed via self._status_ui
        if self._status_ui:
            self._status_ui.update(label="Synthesizing final answer…", state="running", expanded=True)

        try:
            final_response = self._build_compressed_context(state["step_results"])

            # Update UI with completion
            if self._status_ui:
                st.write("\n✅ Synthesis complete.")
                self._status_ui.update(label="Multi-step reasoning complete.", state="complete", expanded=False)

            return {"final_response": final_response}
        except (RuntimeError, ValueError, TypeError) as e:
            LOGGER.exception("Final synthesis failed: %s", e)

            # Update UI with error
            if self._status_ui:
                self._status_ui.update(label="Final synthesis failed.", state="error", expanded=True)

            return {"error_message": f"Final synthesis failed: {e}"}

    def _after_planner(self, state: ReasoningState) -> Literal["continue", "end"]:
        """Route after planning step.

        Returns:
            'continue' if planning succeeded, 'end' if failed or no steps
        """
        if state.get("error_message") or not state.get("planned_steps"):
            return "end"
        return "continue"

    def _step_router(self, state: ReasoningState) -> Literal["search", "refine", "synthesize", "end"]:
        """Route to the next step based on current state.

        Returns:
            Next step type or 'end' if completed/errored
        """
        # Check if we have an error or reached max steps
        if state.get("error_message"):
            return "end"

        current_index = state["current_step_index"]
        planned_steps = state["planned_steps"]

        # Check if we've completed all steps
        if current_index >= len(planned_steps) or current_index >= self._settings.max_steps:
            return "end"

        # Get the next step type
        next_step = planned_steps[current_index]
        if next_step.step_type == StepType.SEARCH:
            return "search"
        if next_step.step_type == StepType.REFINE:
            return "refine"
        if next_step.step_type == StepType.SYNTHESIZE:
            return "synthesize"
        LOGGER.warning("Unknown step type: %s", next_step.step_type)
        return "end"

    async def _summarize_content(self, content: str, query: str) -> str:
        """Summarize content while preserving key information relevant to the query.

        Returns:
            Summarized content or original content if within token limits
        """
        LOGGER.debug("Summarizing content for query: %s", query)
        tokenizer = resources.get_tokenizer()
        token_count = len(tokenizer.encode(content, add_special_tokens=False))

        if token_count <= self._settings.summary_max_tokens:
            return content

        summarization_prompt = templates.SUMMARIZATION_PROMPT.format(
            query=query,
            max_tokens=self._settings.summary_max_tokens,
            content=content,
        )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": summarization_prompt}],
                temperature=self._settings.compression_temperature,
                max_tokens=self._settings.summary_max_tokens,
                seed=self._seed,
            )

            summary = response.choices[0].message.content
            if summary:
                return summary.strip()

            LOGGER.warning("Empty summarization response, using truncated content")
            return content[: self._settings.summary_max_tokens * 2]
        except (RuntimeError, ValueError, TypeError) as e:
            LOGGER.exception("Failed to summarize content: %s", e)
            return content[: self._settings.summary_max_tokens * 2]

    def _build_compressed_context(self, results: list[StepResult]) -> str:
        """Build context with synthesized results first, then recent summaries.

        Returns:
            Compressed context string optimized for token limits
        """
        if not results:
            return ""

        successful = [r for r in results if r.success]
        if not successful:
            return ""

        tokenizer = resources.get_tokenizer()
        max_tokens = self._settings.max_context_tokens

        # Include synthesized results in reverse chronological order
        sections: list[str] = []
        included_in_synthesis: set[int] = set()
        synthesized = [r for r in successful if r.step.step_type == StepType.SYNTHESIZE]

        for idx, result in enumerate(reversed(synthesized), 1):
            block = f"=== Synthesized Knowledge {idx} ===\nQuery: {result.step.query}\nRelevance: {result.relevance_score:.2f}\n{result.content}\n\n"
            potential = "".join(sections) + block
            if len(tokenizer.encode(potential, add_special_tokens=False)) > max_tokens:
                break
            sections.append(block)
            included_in_synthesis.update(result.synthesized_from_indices)

        # Include summaries of recent non-synthesized steps
        n_recent = self._settings.keep_recent_steps_detailed
        if n_recent > 0:
            added = 0
            for i in range(len(successful) - 1, -1, -1):
                if added >= n_recent:
                    break
                result = successful[i]
                if result.step.step_type == StepType.SYNTHESIZE or i in included_in_synthesis:
                    continue

                summary = result.summary or result.content
                block = f"=== Recent Step ({result.step.step_type.value}) ===\nQuery: {result.step.query}\nSummary:\n{summary}\n\n"
                potential = "".join(sections) + block
                if len(tokenizer.encode(potential, add_special_tokens=False)) > max_tokens:
                    break
                sections.append(block)
                added += 1

        compressed = "".join(sections)
        final_tokens = len(tokenizer.encode(compressed, add_special_tokens=False))
        LOGGER.info(
            "Built context in %d tokens (%d synthesized sections, %d recent summaries)",
            final_tokens,
            len(synthesized),
            len(sections) - min(len(sections), len(synthesized)),
        )
        LOGGER.debug("Compressed context: %s", compressed)
        return compressed
