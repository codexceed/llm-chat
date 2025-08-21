#!/usr/bin/env python3
"""Evaluation script for testing the reliability of chatbot.utils.web.lookup_http_urls_in_prompt function.

This script tests the URL fetching functionality with a diverse set of URLs
to measure success rates, response times, and error patterns.
"""

import asyncio
import collections
import dataclasses
import json
import pathlib
import time
from typing import Any

import httpx

import chatbot.web.http

# ruff: noqa: T201


@dataclasses.dataclass
class EvaluationResult:
    """Results from evaluating URL lookup reliability."""

    total_urls: int
    successful_fetches: int
    failed_fetches: int
    success_rate: float
    total_time: float
    avg_time_per_url: float
    error_types: dict[str, int]
    url_results: list[dict[str, Any]]


def load_test_urls(config_file: str = "test_urls.json", test_set: str = "default") -> list[str]:
    """Load test URLs from JSON configuration file.

    Args:
        config_file: pathlib.Path to the JSON configuration file
        test_set: Name of the test set to load

    Returns:
        List of URLs for testing

    Raises:
        ValueError: If the specified test set is not found in the configuration
    """
    config_path = pathlib.Path(__file__).parent / config_file

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    test_sets = config.get("test_sets", {})
    categories = config.get("categories", {})

    if test_set not in test_sets:
        available_sets = list(test_sets.keys())
        raise ValueError(f"Unknown test set '{test_set}'. Available: {available_sets}")

    test_config = test_sets[test_set]
    urls: list[str] = []

    # If test set has direct URLs, use those
    if "urls" in test_config:
        urls.extend(test_config["urls"])

    # If test set references categories, collect URLs from those categories
    if "categories" in test_config:
        for category_name in test_config["categories"]:
            if category_name in categories:
                urls.extend(categories[category_name]["urls"])

    return urls


class WebLookupEvaluator:
    """Evaluates the reliability of web URL lookup functionality."""

    def __init__(self, timeout: float = 10.0, test_urls: list[str] | None = None):
        """Initialize the WebLookupEvaluator.

        Args:
            timeout: Maximum time to wait for HTTP requests
            test_urls: List of URLs to test, defaults to loading from config
        """
        self.timeout = timeout
        self.test_urls = test_urls or load_test_urls()

    async def evaluate_single_prompt(self, prompt: str) -> dict[str, Any]:
        """Evaluate URL lookup for a single prompt.

        Args:
            prompt: Text prompt containing URLs to evaluate

        Returns:
            Dictionary containing evaluation results with keys:
            - prompt: Original prompt text
            - success: Whether content was successfully fetched
            - response_time: Time taken for the request
            - content_length: Total length of fetched content
            - error_type: Type of error if request failed
        """
        start_time = time.time()
        error_type = None
        success = False
        content_length = 0

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                _urls, contents = await chatbot.web.http.fetch_from_http_urls_in_prompt(prompt, client)

                success = len(contents) > 0
                content_length = sum(len(content) for content in contents)

        except httpx.ConnectTimeout:
            error_type = "ConnectTimeout"
        except httpx.ReadTimeout:
            error_type = "ReadTimeout"
        except httpx.ConnectError:
            error_type = "ConnectError"
        except httpx.HTTPStatusError as e:
            error_type = f"HTTPStatusError_{e.response.status_code}"
        except (ValueError, RuntimeError, OSError) as e:
            error_type = f"Other_{type(e).__name__}"

        end_time = time.time()

        return {
            "prompt": prompt,
            "success": success,
            "response_time": end_time - start_time,
            "content_length": content_length,
            "error_type": error_type,
        }

    async def run_evaluation(self) -> EvaluationResult:
        """Run comprehensive evaluation of URL lookup reliability.

        Returns:
            EvaluationResult containing comprehensive test results
        """
        print(f"Starting evaluation with {len(self.test_urls)} URLs...")
        print(f"Timeout: {self.timeout}s")
        print("-" * 60)

        # Create prompts with URLs embedded in text
        prompts = [f"Please analyze this website: {url}" for url in self.test_urls]

        # Run evaluations concurrently
        start_time = time.time()
        tasks = [self.evaluate_single_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Process results
        url_results: list[dict[str, Any]] = []
        url_result: dict[str, Any]
        error_types: dict[str, int] = collections.defaultdict(int)
        successful_fetches = 0
        failed_fetches = 0

        for i, result in enumerate(results):
            if isinstance(result, dict):
                url_result = {"url": self.test_urls[i], **result}
                if result["success"]:
                    successful_fetches += 1
                else:
                    failed_fetches += 1

                if result["error_type"]:
                    error_types[result["error_type"]] += 1
            else:
                url_result = {
                    "url": self.test_urls[i],
                    "success": False,
                    "response_time": 0.0,
                    "content_length": 0,
                    "error_type": f"Exception_{type(result).__name__}",
                }
                failed_fetches += 1
                error_types[url_result["error_type"]] += 1

            url_results.append(url_result)

        success_rate = successful_fetches / len(self.test_urls) * 100
        avg_time_per_url = total_time / len(self.test_urls)

        return EvaluationResult(
            total_urls=len(self.test_urls),
            successful_fetches=successful_fetches,
            failed_fetches=failed_fetches,
            success_rate=success_rate,
            total_time=total_time,
            avg_time_per_url=avg_time_per_url,
            error_types=dict(error_types),
            url_results=url_results,
        )

    def print_results(self, results: EvaluationResult) -> None:
        """Print formatted evaluation results.

        Args:
            results: Evaluation results to print
        """
        print("\n" + "=" * 60)
        print("WEB LOOKUP RELIABILITY EVALUATION RESULTS")
        print("=" * 60)

        print("\nOVERVIEW:")
        print(f"Total URLs tested: {results.total_urls}")
        print(f"Successful fetches: {results.successful_fetches}")
        print(f"Failed fetches: {results.failed_fetches}")
        print(f"Success rate: {results.success_rate:.1f}%")
        print(f"Total evaluation time: {results.total_time:.2f}s")
        print(f"Average time per URL: {results.avg_time_per_url:.2f}s")

        if results.error_types:
            print("\nERROR BREAKDOWN:")
            for error_type, count in sorted(results.error_types.items()):
                percentage = count / results.total_urls * 100
                print(f"  {error_type}: {count} ({percentage:.1f}%)")

        print("\nDETAILED RESULTS:")
        print(f"{'URL':<50} {'Status':<10} {'Time':<8} {'Content':<10} {'Error'}")
        print("-" * 90)

        for result in results.url_results:
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            url_short = result["url"][:47] + "..." if len(result["url"]) > 50 else result["url"]
            content_size = f"{result['content_length']:,}b" if result["content_length"] > 0 else "-"
            error = result["error_type"] or "-"

            print(f"{url_short:<50} {status:<10} {result['response_time']:.2f}s    {content_size:<10} {error}")

        print("\n" + "=" * 60)


async def main() -> None:
    """Main evaluation function."""
    import sys  # pylint: disable=import-outside-toplevel

    # Allow command line argument to select URL set
    url_set = "default"
    if len(sys.argv) > 1:
        url_set = sys.argv[1].lower()

    try:
        test_urls = load_test_urls(test_set=url_set)
        print(f"Using '{url_set.upper()}' test URL set ({len(test_urls)} URLs)")
    except ValueError as e:
        print(f"Error: {e}")
        print("Available test sets: default, minimal, high_reliability, performance")
        return

    evaluator = WebLookupEvaluator(timeout=10.0, test_urls=test_urls)
    results = await evaluator.run_evaluation()
    evaluator.print_results(results)

    # Print reliability assessment
    print("\nRELIABILITY ASSESSMENT:")
    if results.success_rate >= 90:
        print("🟢 EXCELLENT: Function is highly reliable")
    elif results.success_rate >= 75:
        print("🟡 GOOD: Function is mostly reliable with some edge cases")
    elif results.success_rate >= 50:
        print("🟠 FAIR: Function has moderate reliability issues")
    else:
        print("🔴 POOR: Function has significant reliability problems")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if "ConnectTimeout" in results.error_types:
        print("- Consider increasing timeout for slow connections")
    if "HTTPStatusError_404" in results.error_types:
        print("- Add better handling for 404 errors")
    if "ConnectError" in results.error_types:
        print("- Implement retry logic for connection failures")
    if results.avg_time_per_url > 5.0:
        print("- Consider implementing concurrent batching for better performance")


if __name__ == "__main__":
    asyncio.run(main())
