"""Multi-step reasoning orchestrator for complex query processing."""

import asyncio
import datetime
import enum
import json

import httpx
import openai
import pydantic
import scipy.spatial.distance
import streamlit as st
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


class MultiStepOrchestrator:
    """Orchestrates multi-step reasoning chains for complex queries."""

    def __init__(
        self,
        web_context_pipeline: context.WebContextPipeline,
        openai_client: openai.AsyncOpenAI,
        http_client: httpx.AsyncClient,
        model_name: str,
        seed: int,
        reasoning_settings: settings.MultiStepReasoningSettings,
    ) -> None:
        """Initialize the orchestrator.

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

    async def execute_complex_query(self, query: str, status_ui: mutable_status_container.StatusContainer | None = None) -> str | None:
        """Execute a complex query using multi-step reasoning.

        Args:
            query: The complex user query
            status_ui: Optional Streamlit status container for live updates

        Returns:
            Synthesized response from multi-step reasoning, if successfully synthesized. Otherwise, None.
        """
        LOGGER.info("Starting multi-step reasoning for query: %s...", query[:100])

        # Step 1: Plan the reasoning approach
        if status_ui:
            status_ui.update(label="Planning reasoning steps…", state="running", expanded=False)
        plan = await self._plan_reasoning_steps(query)
        # Save for UI visibility
        if not plan.steps:
            LOGGER.warning("No reasoning steps planned.")
            if status_ui:
                status_ui.update(label="No reasoning steps could be planned.", state="error", expanded=True)
            return None

        LOGGER.info("Planned %d reasoning steps", len(plan.steps))
        # Log planned steps in the status container (detailed view)
        # Expand to show live logs while executing
        if status_ui:
            status_ui.update(label=f"Planned {len(plan.steps)} steps. Executing…", state="running", expanded=True)
        st.write("Planned Steps:")
        for i, step in enumerate(plan.steps, 1):
            focus = f" — focus: {step.focus}" if step.focus else ""
            st.write(f"{i}. {step.step_type.value.upper()}: {step.query}{focus}")

        # Step 2: Execute the reasoning chain
        results: list[StepResult] = []
        for i, step in enumerate(plan.steps, 1):
            if i > self._settings.max_steps:
                LOGGER.info("Reached maximum steps (%d), stopping", self._settings.max_steps)
                break

            LOGGER.info("Executing reasoning step %d: %s", i, step.step_type.value)
            if status_ui:
                status_ui.update(label=f"Executing step {i}/{len(plan.steps)}: {step.step_type.value}", expanded=True)
            st.write(f"\n▶ Step {i}: {step.step_type.value.upper()} — query: {step.query}")

            try:
                result = await asyncio.wait_for(self._execute_step(step, results), timeout=self._settings.step_timeout)
                if result.success:
                    result.summary = await self._summarize_content(result.content, step.query)
                results.append(result)

                if not result.success:
                    LOGGER.warning("Step %d failed: %s", i, result.error_message)
                    if status_ui:
                        st.write(f"❌ Step {i} failed: {result.error_message}")
                        status_ui.update(label=f"Step {i} failed", state="error", expanded=True)
                    break
                if status_ui:
                    synth = " (SYNTH)" if step.step_type == StepType.SYNTHESIZE else ""
                    st.write(f"✅ Completed step {i}: {step.step_type.value}{synth}")
            except asyncio.TimeoutError:
                LOGGER.exception("Step %d timed out", i)
                if status_ui:
                    st.write(f"⏱️ Step {i} timed out")
                    status_ui.update(label=f"Step {i} timed out", state="error", expanded=True)
                break
            except (ValueError, TypeError, RuntimeError) as e:
                LOGGER.exception("Step %d failed with error: %s", i, e)
                if status_ui:
                    st.write(f"❌ Step {i} error: {e}")
                    status_ui.update(label=f"Error at step {i}", state="error", expanded=True)
                break

        # Step 3: Synthesize final response
        if results:
            if status_ui:
                status_ui.update(label="Synthesizing final answer…", state="running", expanded=True)
            synthesized = await self._synthesize_reasoned_context(results)
            if status_ui:
                st.write("\n✅ Synthesis complete.")
            return synthesized

        LOGGER.warning("No successful execution of reasoning steps.")
        if status_ui:
            status_ui.update(label="No successful steps executed.", state="error", expanded=True)
        return None

    async def _plan_reasoning_steps(self, query: str) -> ReasoningPlanSchema:
        """Plan the reasoning steps for a complex query.

        Args:
            query: User query

        Returns:
            List of planned reasoning steps
        """
        timestamp = datetime.datetime.now().isoformat()
        planning_prompt = templates.PLANNING_PROMPT.format(
            timestamp=timestamp,
            max_steps=self._settings.max_steps,
            query=query,
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
                return ReasoningPlanSchema(steps=[])

            plan_content = response.choices[0].message.parsed

            if plan_content is None:
                LOGGER.error("Planning LLM returned no parsed content")
                return ReasoningPlanSchema(steps=[])

            LOGGER.debug("Structured planning response: %s", plan_content.model_dump_json(indent=2))

            return plan_content

        except (ValueError, TypeError, json.JSONDecodeError, pydantic.ValidationError) as e:
            LOGGER.exception("Structured planning failed: %s", e)
            return ReasoningPlanSchema(steps=[])

    async def _execute_step(self, step: ReasoningStep, previous_results: list[StepResult]) -> StepResult:
        """Execute a single reasoning step.

        Args:
            step: The step to execute
            previous_results: Results from previous steps

        Returns:
            Result of step execution
        """
        try:
            synthesized_indices: list[int] = []
            if step.step_type == StepType.SEARCH:
                content = await self._execute_search_step(step)
            elif step.step_type == StepType.REFINE:
                content = await self._execute_refine_step(step, previous_results)
            elif step.step_type == StepType.SYNTHESIZE:
                content, synthesized_indices = await self._execute_synthesize_step(previous_results)
            else:
                error_message = f"Unknown step type: {step.step_type}"
                return StepResult(
                    step=step,
                    content="",
                    success=False,
                    error_message=error_message,
                    query=step.query,
                )

            return StepResult(
                step=step,
                content=content,
                success=True,
                query=step.query,
                synthesized_from_indices=synthesized_indices,
            )

        except (ValueError, TypeError, RuntimeError) as e:
            LOGGER.exception("Step execution failed: %s", e)
            return StepResult(
                step=step,
                content="",
                success=False,
                error_message=str(e),
                query=step.query,
            )

    async def _execute_search_step(self, step: ReasoningStep) -> str:
        """Execute a search step.

        Args:
            step: The search step to execute

        Returns:
            Search results content.
        """
        LOGGER.info('Executing search step with query: "%s"', step.query)

        # Use the existing web context pipeline for search
        web_context_dict = await self._web_pipeline.gather_web_context(
            step.query,
            self._http_client,
            enable_web_search=True,
            force_web_search=True,  # Force search for reasoning steps
            search_num_results=self._settings.search_top_k,
        )

        # Extract search results
        search_content = web_context_dict.get("web_search", "")
        if not search_content:
            LOGGER.warning("No search results obtained")

        return search_content

    async def _execute_refine_step(self, step: ReasoningStep, previous_results: list[StepResult]) -> str:
        """Execute a refinement step based on previous results.

        Returns:
            Refined search results content.

        Raises:
            ValueError: If refinement LLM returns empty response.
        """
        LOGGER.info("Executing refine step with focus: %s", step.focus)

        if any(r.summary is None for r in previous_results if r.success):
            raise ValueError("Some previous results lack summaries, which may affect refinement quality.")

        # Create a refined query based on previous results and current focus
        # Use summaries for compression if enabled
        if self._settings.enable_context_compression:
            previous_content = "\n\n".join([r.summary or "" for r in previous_results if r.success])
        else:
            previous_content = "\n\n".join([r.content for r in previous_results if r.success])

        refine_prompt = templates.REFINE_PROMPT.format(
            focus=step.focus,
            query=step.query,
            previous_content=previous_content,
        )
        LOGGER.debug("Refinement prompt: %s", refine_prompt)

        # Get refined query from LLM
        response = await self._openai_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        refined_query_content = response.choices[0].message.content or ""
        if not refined_query_content:
            LOGGER.error("Empty response from refinement LLM")
        refined_query = refined_query_content.strip()
        LOGGER.debug("Refined query: %s", refined_query)

        # Execute search with refined query
        refined_step = ReasoningStep(step_type=StepType.SEARCH, query=refined_query, focus=step.focus)

        return await self._execute_search_step(refined_step)

    async def _execute_synthesize_step(self, previous_results: list[StepResult]) -> tuple[str, list[int]]:
        """Execute a synthesis step to combine previous results.

        Only includes results that haven't already been synthesized to avoid redundancy.
        Synthesized results already contain cumulative knowledge from their prior steps.

        Returns:
            Tuple of (combined content, indices of results that were synthesized)

        Raises:
            ValueError: If no successful results are available to synthesize.
        """
        LOGGER.info("Executing synthesize step")

        # Find which results have already been incorporated into synthesized steps
        already_synthesized = set()
        for result in previous_results:
            if result.step.step_type == StepType.SYNTHESIZE and result.success:
                already_synthesized.update(result.synthesized_from_indices)

        # Only include results that haven't been synthesized yet, plus any synthesized results
        combined_content = ""
        synthesized_indices = []
        for i, result in enumerate(previous_results):
            if result.success:
                # Include if it's a synthesized result OR if it hasn't been synthesized yet
                if result.step.step_type == StepType.SYNTHESIZE or i not in already_synthesized:
                    combined_content += f"=== Step {i + 1} Results ===\n{result.content}\n\n"
                    synthesized_indices.append(i)
                else:
                    LOGGER.debug("Skipping already-synthesized result at index %d", i)

        if not combined_content:
            error_msg = "No successful results to synthesize"
            raise ValueError(error_msg)

        LOGGER.info(
            "Synthesizing from %d results (skipped %d already-synthesized)",
            len(synthesized_indices),
            len([r for r in previous_results if r.success]) - len(synthesized_indices),
        )

        return combined_content, synthesized_indices

    async def _synthesize_reasoned_context(self, results: list[StepResult]) -> str:
        """Synthesize reasoned context from all step results.

        Args:
            results: List of step results

        Returns:
            Synthesized reasoning context string
        """
        LOGGER.info("Synthesizing final response from %d steps", len(results))

        # Always use intelligent context management (compression logic now in StepResult)
        return self._build_compressed_context(results)

    async def _summarize_content(self, content: str, query: str) -> str:
        """Summarize content while preserving key information relevant to the query.

        Args:
            content: Content to summarize
            query: Original query for context

        Returns:
            Summarized content
        """
        tokenizer = resources.get_tokenizer()
        token_count = len(tokenizer.encode(content, add_special_tokens=False))

        if token_count <= self._settings.summary_max_tokens:
            return content  # Already within limits

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
            return content[: self._settings.summary_max_tokens * 2]  # Rough character estimate
        except (ValueError, TypeError, RuntimeError) as e:
            LOGGER.exception("Failed to summarize content: %s", e)
            return content[: self._settings.summary_max_tokens * 2]  # Fallback to truncation

    def _build_compressed_context(self, results: list[StepResult]) -> str:
        """Build context with synthesized results first, then recent summaries.

        - Prioritize synthesized steps (they subsume earlier content).
        - Then include summaries of the last N non-synthesized steps that were not
          already incorporated by an included synthesis.
        - Use `summary_max_tokens` produced earlier; do not add extra caps.

        Args:
            results: List of step results

        Returns:
            Compressed context string
        """
        if not results:
            return ""

        successful = [r for r in results if r.success]
        if not successful:
            return ""

        tokenizer = resources.get_tokenizer()
        max_tokens = self._settings.max_context_tokens

        # 1) Include synthesized results in reverse chronological order
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

        # 2) Include summaries of the last N non-synth steps not already subsumed
        n_recent = self._settings.keep_recent_steps_detailed
        if n_recent > 0:
            added = 0
            # Walk back through overall results for true recency
            for i in range(len(successful) - 1, -1, -1):
                if added >= n_recent:
                    break
                result = successful[i]
                if result.step.step_type == StepType.SYNTHESIZE:
                    continue
                if i in included_in_synthesis:
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
        LOGGER.debug("Compressed context (%d tokens):\n%s", final_tokens, compressed)
        return compressed
