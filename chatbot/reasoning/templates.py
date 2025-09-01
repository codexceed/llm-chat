"""Prompt templates for the multi-step reasoning orchestrator."""

PLANNING_PROMPT = """
Date: {timestamp}
You are a reasoning planner. Given a complex user query, plan the minimum number of steps needed to answer it thoroughly.

Available step types:
- SEARCH: Perform web search with a specific query
- REFINE: Refine search based on previous results with a focused query
- SYNTHESIZE: Combine findings to answer the original question. Synthesized results take priority over constituent parts in the final response.

Planning guidelines:
- Use SYNTHESIZE strategically to consolidate knowledge at logical breakpoints
- Multiple synthesis steps can build upon each other hierarchically
- Recent synthesized results contain the most refined, cumulative knowledge
- Plan the minimum steps needed while ensuring comprehensive coverage
- Plan for no more than {max_steps} steps

User Query: {query}
"""

REFINE_PROMPT = """
Based on the previous search results, perform a more focused search.

Focus: {focus}
Query: {query}

Previous results summary:
{previous_content}

Provide a refined search query that will help get more specific information about the focus area.
"""

SUMMARIZATION_PROMPT = """Summarize the following content while preserving key information relevant to the query "{query}".

Focus on:
1. Main findings and facts
2. Specific data, numbers, and dates
3. Key relationships and connections
4. Actionable insights

Keep the summary concise but comprehensive, under {max_tokens} tokens.

Content:
{content}

Summary:
"""
