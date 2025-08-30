"""Query complexity classifier for routing between simple and complex reasoning."""

import enum
import re

from streamlit import logger

from chatbot import settings

LOGGER = logger.get_logger("streamlit")


class QueryComplexity(enum.Enum):
    """Query complexity levels."""

    SIMPLE = "simple"
    COMPLEX = "complex"


class QueryComplexityClassifier:
    """Classifies query complexity to route between context injection and multi-step reasoning."""

    def __init__(self) -> None:
        """Initialize the classifier with settings."""
        self._complex_keywords = settings.CHATBOT_SETTINGS.multi_step.complex_query_keywords
        self._multi_entity_threshold = settings.CHATBOT_SETTINGS.multi_step.multi_entity_threshold

    def classify_query(self, query: str) -> QueryComplexity:
        """Classify a query as simple or complex.

        Args:
            query: User query string

        Returns:
            QueryComplexity indicating routing decision
        """
        LOGGER.info("Classifying query complexity for: %s", query[:100])

        # Check for complexity keywords
        if self._has_complexity_keywords(query):
            LOGGER.info("Query classified as COMPLEX due to complexity keywords")
            return QueryComplexity.COMPLEX

        # Check for multiple entities (indicates comparative/multi-domain queries)
        if self._has_multiple_entities(query):
            LOGGER.info("Query classified as COMPLEX due to multiple entities")
            return QueryComplexity.COMPLEX

        # Check for question patterns that indicate multi-step reasoning
        if self._has_complex_question_patterns(query):
            LOGGER.info("Query classified as COMPLEX due to complex question patterns")
            return QueryComplexity.COMPLEX

        LOGGER.info("Query classified as SIMPLE")
        return QueryComplexity.SIMPLE

    def _has_complexity_keywords(self, query: str) -> bool:
        """Check if query contains complexity keywords.

        Returns:
            True if query contains complexity keywords, False otherwise.
        """
        query_lower = query.lower()

        for keyword in self._complex_keywords:
            if keyword in query_lower:
                LOGGER.debug("Found complexity keyword: %s", keyword)
                return True

        return False

    def _has_multiple_entities(self, query: str) -> bool:
        """Check if query contains multiple named entities or topics.

        Returns:
            True if query contains multiple entities above threshold, False otherwise.
        """
        # Simple heuristic: look for proper nouns and company/product patterns

        # Pattern for capitalized words (potential proper nouns)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)

        # Pattern for common entity types
        entity_patterns = [
            r"\b(?:iPhone|Samsung|Google|Apple|Microsoft|Amazon|Tesla|OpenAI|Anthropic)\b",
            r"\b(?:Company|Inc|Corp|Ltd|LLC)\b",
            r"\b[A-Z]{2,}\b",  # Acronyms
        ]

        entities = set()

        # Add proper nouns but exclude common question words
        for noun in proper_nouns:
            if noun not in {
                "What",
                "How",
                "Why",
                "When",
                "Where",
                "Which",
                "Who",
                "Will",
                "Can",
                "The",
                "This",
                "That",
            }:
                entities.add(noun)

        # Add entities from patterns
        for pattern in entity_patterns:
            entities.update(re.findall(pattern, query, re.IGNORECASE))

        # Remove common words that aren't entities
        common_words = {
            "Today",
            "Latest",
            "News",
            "Recent",
            "Current",
            "New",
            "Update",
            "Information",
            "Install",
            "Weather",
            "President",
            "About",
            "For",
            "In",
            "On",
            "At",
            "To",
            "From",
            "With",
            "By",
        }
        entities -= common_words

        entity_count = len(entities)
        LOGGER.debug("Found %d entities: %s", entity_count, list(entities))

        return entity_count >= self._multi_entity_threshold

    @staticmethod
    def _has_complex_question_patterns(query: str) -> bool:
        """Check for question patterns that indicate multi-step reasoning needs.

        Returns:
            True if query contains complex question patterns, False otherwise.
        """
        query_lower = query.lower()

        # Patterns that suggest multi-step reasoning
        complex_patterns = [
            r"\bhow.*?relate.*?to\b",  # "how does X relate to Y"
            r"\bwhat.*?impact.*?on\b",  # "what impact does X have on Y"
            r"\bwhy.*?caused.*?by\b",  # "why was X caused by Y"
            r"\bwhat.*?led.*?to\b",  # "what led to X"
            r"\bhow.*?different.*?from\b",  # "how is X different from Y"
            r"\bwhat.*?happens.*?if\b",  # "what happens if X"
            r"\bhow.*?affect.*?future\b",  # "how will X affect future"
            r"\b(?:first|then|next|finally)\b.*?\b(?:first|then|next|finally)\b",  # Sequential indicators
            r"\bstep.*?by.*?step\b",  # "step by step"
            r"\bchronological\b",  # Timeline requests
        ]

        for pattern in complex_patterns:
            if re.search(pattern, query_lower):
                LOGGER.debug("Found complex question pattern: %s", pattern)
                return True

        return False
