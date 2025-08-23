"""Multi-step reasoning orchestrator for complex query processing."""

import asyncio
import dataclasses
import enum
import json

import httpx
import openai
from streamlit import logger

from chatbot.web import context

LOGGER = logger.get_logger(__name__)


class StepType(enum.Enum):
    """Types of reasoning steps."""

    SEARCH = "search"
    REFINE = "refine"
    SYNTHESIZE = "synthesize"


@dataclasses.dataclass
class ReasoningStep:
    """Represents a single reasoning step."""

    step_type: StepType
    query: str
    focus: str = ""
    rationale: str = ""


@dataclasses.dataclass
class StepResult:
    """Result from executing a reasoning step."""

    step: ReasoningStep
    content: str
    success: bool
    error_message: str = ""


class MultiStepOrchestrator:
    """Orchestrates multi-step reasoning chains for complex queries."""

    def __init__(
        self,
        web_context_pipeline: context.WebContextPipeline,
        openai_client: openai.OpenAI,
        http_client: httpx.AsyncClient,
        model_name: str,
        seed: int | None = None,
        max_steps: int = 5,
        planning_temperature: float = 0.3,
        step_timeout: int = 30,
        search_top_k: int = 5,
        max_reasoning_tokens: int = 2000,
        max_context_tokens: int = 4000,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            web_context_pipeline: Pipeline for web context processing
            openai_client: OpenAI client for LLM interactions
            http_client: HTTP client for web requests
            model_name: LLM model name to use
            seed: Optional random seed for reproducibility
            max_steps: Maximum reasoning steps to execute
            planning_temperature: Temperature for planning LLM
            step_timeout: Timeout in seconds for each reasoning step
            search_top_k: Number of search results to fetch for search steps
            max_reasoning_tokens: Maximum tokens for reasoning LLM responses
            max_context_tokens: Maximum tokens for overall context length
        """
        self._web_pipeline = web_context_pipeline
        self._openai_client = openai_client
        self._http_client = http_client
        self._model = model_name
        self._max_steps = max_steps
        self._planning_temperature = planning_temperature
        self._step_timeout = step_timeout
        self._seed = seed
        self._search_top_k = search_top_k
        self._max_reasoning_tokens = max_reasoning_tokens
        self._max_context_tokens = max_context_tokens

    async def execute_complex_query(self, query: str) -> str | None:
        """Execute a complex query using multi-step reasoning.

        Args:
            query: The complex user query

        Returns:
            Synthesized response from multi-step reasoning, if successfully synthesized. Otherwise, None.
        """
        LOGGER.info("Starting multi-step reasoning for query: %s", query[:100])

        # Step 1: Plan the reasoning approach
        steps = await self._plan_reasoning_steps(query)
        if not steps:
            LOGGER.warning("No reasoning steps planned.")
            return None

        LOGGER.info("Planned %d reasoning steps", len(steps))

        # Step 2: Execute the reasoning chain
        results: list[StepResult] = []
        for i, step in enumerate(steps, 1):
            if i > self._max_steps:
                LOGGER.info("Reached maximum steps (%d), stopping", self._max_steps)
                break

            LOGGER.info("Executing step %d: %s", i, step.step_type.value)

            try:
                result = await asyncio.wait_for(self._execute_step(step, results), timeout=self._step_timeout)
                results.append(result)

                if not result.success:
                    LOGGER.warning("Step %d failed: %s", i, result.error_message)
                    break

                # Check if we should stop early
                if self._should_stop_reasoning(results, query):
                    LOGGER.info("Stopping reasoning chain early at step %d", i)
                    break

            except asyncio.TimeoutError:
                LOGGER.exception("Step %d timed out", i)
                break
            except (ValueError, TypeError, RuntimeError) as e:
                LOGGER.exception("Step %d failed with error: %s", i, e)
                break

        # Step 3: Synthesize final response
        if results:
            return await self._synthesize_reasoned_context(results)

        LOGGER.warning("No successful execution of reasoning steps.")
        return None

    async def _plan_reasoning_steps(self, query: str) -> list[ReasoningStep]:
        """Plan the reasoning steps for a complex query.

        Args:
            query: User query

        Returns:
            List of planned reasoning steps
        """
        planning_prompt = f"""
You are a reasoning planner. Given a complex user query, plan the steps needed to answer it thoroughly.

Available step types:
- SEARCH: Perform web search with a specific query
- REFINE: Refine search based on previous results with a focused query
- SYNTHESIZE: Combine findings to answer the original question

Provide a JSON list of steps with this format:
[
  {{
    "step_type": "SEARCH",
    "query": "specific search query",
    "focus": "what to focus on in results",
    "rationale": "why this step is needed"
  }}
]

Keep it to maximum 3 steps. Focus on the most important information needed.

User Query: {query}

Reasoning Plan:"""

        try:
            response = self._openai_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=self._planning_temperature,
                max_tokens=800,
                seed=self._seed,
            )

            plan_text = response.choices[0].message.content
            if not plan_text:
                LOGGER.error("Empty response from planning LLM")
                return []

            LOGGER.debug("Raw planning response: %s", plan_text)

            # Extract JSON from response
            json_start = plan_text.find("[")
            json_end = plan_text.rfind("]") + 1

            if json_start == -1 or json_end == 0:
                LOGGER.error("No valid JSON found in planning response")
                return []

            plan_json = plan_text[json_start:json_end]
            steps_data = json.loads(plan_json)

            steps = []
            invalid_steps = []

            for step_data in steps_data:
                # Validate required fields first
                if not isinstance(step_data, dict) or "step_type" not in step_data or "query" not in step_data:
                    invalid_steps.append((step_data, "Missing required fields"))
                    continue

                try:
                    step_type_value = step_data["step_type"].lower()
                    if step_type_value not in [s.value for s in StepType]:
                        invalid_steps.append((step_data, f"Invalid step type: {step_type_value}"))
                        continue

                    step = ReasoningStep(
                        step_type=StepType(step_type_value),
                        query=step_data["query"],
                        focus=step_data.get("focus", ""),
                        rationale=step_data.get("rationale", ""),
                    )
                    steps.append(step)
                except (KeyError, ValueError) as e:
                    invalid_steps.append((step_data, str(e)))

            # Log all invalid steps at once
            if invalid_steps:
                for step_data, error in invalid_steps:
                    LOGGER.error("Invalid step data: %s, error: %s", step_data, error)

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            LOGGER.exception("Planning failed: %s", e)
            return []

        return steps

    async def _execute_step(self, step: ReasoningStep, previous_results: list[StepResult]) -> StepResult:
        """Execute a single reasoning step.

        Args:
            step: The step to execute
            previous_results: Results from previous steps

        Returns:
            Result of step execution
        """
        try:
            if step.step_type == StepType.SEARCH:
                content = await self._execute_search_step(step)
            elif step.step_type == StepType.REFINE:
                content = await self._execute_refine_step(step, previous_results)
            elif step.step_type == StepType.SYNTHESIZE:
                content = await self._execute_synthesize_step(step, previous_results)
            else:
                error_message = f"Unknown step type: {step.step_type}"
                return StepResult(
                    step=step,
                    content="",
                    success=False,
                    error_message=error_message,
                )

            return StepResult(step=step, content=content, success=True)

        except (ValueError, TypeError, RuntimeError) as e:
            LOGGER.exception("Step execution failed: %s", e)
            return StepResult(step=step, content="", success=False, error_message=str(e))

    async def _execute_search_step(self, step: ReasoningStep) -> str:
        """Execute a search step.

        Returns:
            Search results content.

        Raises:
            ValueError: If no search results are obtained.
        """
        LOGGER.info("Executing search step with query: %s", step.query)

        # Use the existing web context pipeline for search
        web_context_dict = await self._web_pipeline.gather_web_context(
            step.query,
            self._http_client,
            enable_web_search=True,
            force_web_search=True,  # Force search for reasoning steps
            search_num_results=self._search_top_k,
        )

        # Extract search results
        search_content = web_context_dict.get("web_search", "")
        if not search_content:
            error_msg = "No search results obtained"
            raise ValueError(error_msg)

        return search_content

    async def _execute_refine_step(self, step: ReasoningStep, previous_results: list[StepResult]) -> str:
        """Execute a refinement step based on previous results.

        Returns:
            Refined search results content.

        Raises:
            ValueError: If refinement LLM returns empty response.
        """
        LOGGER.info("Executing refine step with focus: %s", step.focus)

        # Create a refined query based on previous results and current focus
        previous_content = "\n\n".join([r.content[:500] for r in previous_results if r.success])

        refine_prompt = f"""
Based on the previous search results, perform a more focused search.

Focus: {step.focus}
Query: {step.query}

Previous results summary:
{previous_content[:1000]}

Provide a refined search query that will help get more specific information about the focus area.
"""

        # Get refined query from LLM
        response = self._openai_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        refined_query_content = response.choices[0].message.content
        if not refined_query_content:
            error_msg = "Empty response from refinement LLM"
            raise ValueError(error_msg)
        refined_query = refined_query_content.strip()
        LOGGER.debug("Refined query: %s", refined_query)

        # Execute search with refined query
        refined_step = ReasoningStep(step_type=StepType.SEARCH, query=refined_query, focus=step.focus)

        return await self._execute_search_step(refined_step)

    async def _execute_synthesize_step(self, _step: ReasoningStep, previous_results: list[StepResult]) -> str:
        """Execute a synthesis step to combine previous results.

        Returns:
            Combined content from all previous successful results.

        Raises:
            ValueError: If no successful results are available to synthesize.
        """
        LOGGER.info("Executing synthesize step")

        # Combine all successful results
        combined_content = ""
        for i, result in enumerate(previous_results, 1):
            if result.success:
                combined_content += f"=== Step {i} Results ===\n{result.content}\n\n"

        if not combined_content:
            error_msg = "No successful results to synthesize"
            raise ValueError(error_msg)

        return combined_content

    def _should_stop_reasoning(self, results: list[StepResult], original_query: str) -> bool:
        """Determine if reasoning chain should stop early.

        Args:
            results: Results so far
            original_query: Original user query

        Returns:
            True if reasoning should stop
        """
        if not results:
            return False

        # Simple heuristic: stop if we have enough content
        total_content_length = sum(len(r.content) for r in results if r.success)

        # Stop if we have substantial content (adaptive based on query length)
        query_length = len(original_query)
        target_content_length = max(self._max_context_tokens, query_length * 10)

        return total_content_length >= target_content_length

    async def _synthesize_reasoned_context(self, results: list[StepResult]) -> str:
        """Synthesize reasoned context from all step results.

        Args:
            results: List of step results

        Returns:
            Synthesized reasoning context string
        """
        LOGGER.info("Synthesizing final response from %d steps", len(results))

        # Combine all successful results
        reasoning_context = ""
        for i, result in enumerate(results, 1):
            if result.success:
                reasoning_context += f"=== Reasoning Step {i}: {result.step.step_type.value} ===\n"
                reasoning_context += f"Query: {result.step.query}\n"
                if result.step.focus:
                    reasoning_context += f"Focus: {result.step.focus}\n"
                reasoning_context += f"Results:\n{result.content}\n\n"

        return reasoning_context
