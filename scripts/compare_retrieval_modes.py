#!/usr/bin/env python3
"""Demonstration script comparing Dense vs Hybrid retrieval performance.

This script shows the practical differences between:
1. Dense-only retrieval (semantic similarity via embeddings)
2. Hybrid retrieval (dense + BM42 sparse embeddings)

The script will:
- Create sample documents with different characteristics
- Index them using both retrieval modes
- Run various queries to demonstrate retrieval differences
- Compare results and highlight when hybrid performs better
"""

import json
import logging
import pathlib
import sys
import time
from typing import Any

import llama_index.core
import qdrant_client
from llama_index import core
from llama_index.embeddings import huggingface
from llama_index.vector_stores import qdrant as qdrant_vector_store
from qdrant_client.http import models

# Add project root to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ruff: noqa: T201


def load_sample_documents() -> list[dict[str, Any]]:
    """Load sample documents from JSON file.

    Returns:
        List of document dictionaries containing title and content

    Raises:
        FileNotFoundError: If the sample documents file doesn't exist
        JSONDecodeError: If the JSON file is malformed
    """
    data_dir = pathlib.Path(__file__).parent / "data"
    documents_file = data_dir / "sample_documents.json"

    try:
        with open(documents_file, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Sample documents file not found: {documents_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing sample documents JSON: {e}")
        raise


def load_test_queries() -> list[dict[str, Any]]:
    """Load test queries from JSON file.

    Returns:
        List of test query dictionaries with query, description, and expected results

    Raises:
        FileNotFoundError: If the test queries file doesn't exist
        JSONDecodeError: If the JSON file is malformed
    """
    data_dir = pathlib.Path(__file__).parent / "data"
    queries_file = data_dir / "test_queries.json"

    try:
        with open(queries_file, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Test queries file not found: {queries_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing test queries JSON: {e}")
        raise


class RetrievalComparer:
    """Compares dense vs hybrid retrieval performance."""

    def __init__(self) -> None:
        self.embedding_model: huggingface.HuggingFaceEmbedding = huggingface.HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", device="cpu"
        )
        self.client: qdrant_client.QdrantClient
        self.dense_index: llama_index.core.VectorStoreIndex
        self.hybrid_index: llama_index.core.VectorStoreIndex

        # Configure LlamaIndex for larger, complex documents
        core.Settings.embed_model = self.embedding_model
        core.Settings.chunk_size = 1024
        core.Settings.chunk_overlap = 100

        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(url="http://localhost:6333")

        self._create_indexes()

    def setup_collections(self) -> None:
        """Create separate collections for dense and hybrid testing."""
        collections_to_create: list[tuple[str, bool]] = [
            ("dense_test", False),  # Dense only
            ("hybrid_test", True),  # Hybrid with sparse
        ]

        for collection_name, _use_sparse in collections_to_create:
            try:
                # Delete existing collection if it exists
                try:
                    self.client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass

                # Create new collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # BGE-small embedding dimension
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {collection_name}")

            except Exception as e:  # noqa: PERF203
                logger.error(f"Error setting up collection {collection_name}: {e}")
                raise

    def _create_indexes(self) -> None:
        """Create dense and hybrid vector store indexes."""
        # Dense-only vector store
        dense_vector_store = qdrant_vector_store.QdrantVectorStore(client=self.client, collection_name="dense_test")

        # Hybrid vector store with BM42 sparse model
        hybrid_vector_store = qdrant_vector_store.QdrantVectorStore(
            client=self.client,
            collection_name="hybrid_test",
            fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
        )

        # Load data from JSON files
        sample_documents: list[dict[str, Any]] = load_sample_documents()

        # Create documents
        documents: list[llama_index.core.Document] = [
            llama_index.core.Document(text=f"{doc['title']}\n\n{doc['content']}", metadata={"title": doc["title"]})
            for doc in sample_documents
        ]

        # Create indexes
        logger.info("Creating dense-only index...")
        self.dense_index = llama_index.core.VectorStoreIndex.from_documents(documents, vector_store=dense_vector_store)

        logger.info("Creating hybrid index...")
        self.hybrid_index = llama_index.core.VectorStoreIndex.from_documents(
            documents, vector_store=hybrid_vector_store
        )

        logger.info("Both indexes created successfully!")

    def run_comparison(self) -> None:
        """Run retrieval comparison for all test queries."""
        dense_retriever = self.dense_index.as_retriever(similarity_top_k=5)
        hybrid_retriever = self.hybrid_index.as_retriever(similarity_top_k=5, sparse_top_k=5)

        # Load test queries from JSON
        test_queries: list[dict[str, Any]] = load_test_queries()
        results_summary: list[dict[str, Any]] = []

        logger.info(f"Running comparison with {len(test_queries)} test queries...")

        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]
            expected_best = test_case["expected_best"]
            category = test_case.get("category", "unknown")

            logger.info(f"\nTest {i}/{len(test_queries)}: {query}")
            logger.info(f"Category: {category} - {description}")
            logger.info(f"Expected best match: {expected_best}")

            # Dense retrieval
            start_time = time.time()
            dense_results = dense_retriever.retrieve(query)
            dense_time = time.time() - start_time

            dense_titles: list[str] = []
            for _j, node in enumerate(dense_results, 1):
                title: str = node.metadata.get("title", "Unknown")
                getattr(node, "score", "N/A")
                dense_titles.append(title)

            # Hybrid retrieval
            start_time = time.time()
            hybrid_results = hybrid_retriever.retrieve(query)
            hybrid_time = time.time() - start_time

            hybrid_titles: list[str] = []
            for _j, node in enumerate(hybrid_results, 1):
                hybrid_title: str = node.metadata.get("title", "Unknown")
                getattr(node, "score", "N/A")
                hybrid_titles.append(hybrid_title)

            # Analysis
            dense_found_best: bool = expected_best in dense_titles
            hybrid_found_best: bool = expected_best in hybrid_titles

            if dense_found_best and hybrid_found_best:
                dense_rank = dense_titles.index(expected_best) + 1
                hybrid_rank = hybrid_titles.index(expected_best) + 1
                if hybrid_rank < dense_rank:
                    winner = "🏆 HYBRID WINS - Better ranking"
                elif dense_rank < hybrid_rank:
                    winner = "🏆 DENSE WINS - Better ranking"
                else:
                    winner = "🤝 TIE - Same ranking"
            elif hybrid_found_best and not dense_found_best:
                winner = "🏆 HYBRID WINS - Found expected result"
            elif dense_found_best and not hybrid_found_best:
                winner = "🏆 DENSE WINS - Found expected result"
            else:
                winner = "❌ BOTH MISSED - Neither found expected result"

            results_summary.append(
                {
                    "query": query,
                    "category": category,
                    "description": description,
                    "expected": expected_best,
                    "dense_found": dense_found_best,
                    "hybrid_found": hybrid_found_best,
                    "winner": winner,
                    "dense_time": dense_time,
                    "hybrid_time": hybrid_time,
                    "dense_titles": dense_titles,
                    "hybrid_titles": hybrid_titles,
                }
            )

        # Final summary
        self.print_final_summary(results_summary)

    def print_final_summary(self, results: list[dict[str, Any]]) -> None:
        """Print comprehensive comparison summary with detailed analysis."""
        print("\n" + "=" * 80)
        print("🏁 FINAL RETRIEVAL COMPARISON SUMMARY")
        print("=" * 80)

        # Overall statistics
        dense_wins = sum(1 for r in results if "DENSE WINS" in r["winner"])
        hybrid_wins = sum(1 for r in results if "HYBRID WINS" in r["winner"])
        ties = sum(1 for r in results if "TIE" in r["winner"])
        both_missed = sum(1 for r in results if "BOTH MISSED" in r["winner"])

        total_tests = len(results)
        avg_dense_time = sum(r["dense_time"] for r in results) / total_tests
        avg_hybrid_time = sum(r["hybrid_time"] for r in results) / total_tests

        print(f"\n📊 OVERALL RESULTS ({total_tests} test queries):")
        print(f"  🏆 Hybrid Wins: {hybrid_wins} ({hybrid_wins / total_tests * 100:.1f}%)")
        print(f"  🏆 Dense Wins: {dense_wins} ({dense_wins / total_tests * 100:.1f}%)")
        print(f"  🤝 Ties: {ties} ({ties / total_tests * 100:.1f}%)")
        print(f"  ❌ Both Missed: {both_missed} ({both_missed / total_tests * 100:.1f}%)")

        # Performance analysis
        print("\n⏱️  PERFORMANCE METRICS:")
        print(f"  Dense Average Time: {avg_dense_time:.4f}s")
        print(f"  Hybrid Average Time: {avg_hybrid_time:.4f}s")

        if avg_hybrid_time > avg_dense_time:
            slowdown = ((avg_hybrid_time - avg_dense_time) / avg_dense_time) * 100
            print(f"  Hybrid Slowdown: {slowdown:.1f}%")
        else:
            speedup = ((avg_dense_time - avg_hybrid_time) / avg_dense_time) * 100
            print(f"  Hybrid Speedup: {speedup:.1f}%")

        # Success rate analysis
        dense_success_rate = (dense_wins + ties) / total_tests * 100
        hybrid_success_rate = (hybrid_wins + ties) / total_tests * 100

        print("\n✅ SUCCESS RATES (Found Expected Result):")
        print(f"  Dense Success Rate: {dense_success_rate:.1f}%")
        print(f"  Hybrid Success Rate: {hybrid_success_rate:.1f}%")

        # Category-based analysis using JSON categories
        categories: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            category: str = result.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        print("\n🏷️  PERFORMANCE BY CATEGORY:")
        for category, queries in categories.items():
            if queries:
                cat_hybrid_wins = sum(1 for r in queries if "HYBRID WINS" in r["winner"])
                cat_dense_wins = sum(1 for r in queries if "DENSE WINS" in r["winner"])
                cat_ties = sum(1 for r in queries if "TIE" in r["winner"])
                cat_total = len(queries)

                category_name = category.replace("_", " ").title()
                print(f"  {category_name} ({cat_total} queries):")
                print(f"    Hybrid: {cat_hybrid_wins}, Dense: {cat_dense_wins}, Ties: {cat_ties}")

                if cat_hybrid_wins > cat_dense_wins:
                    print(f"    → 🏆 HYBRID ADVANTAGE in {category_name}")
                elif cat_dense_wins > cat_hybrid_wins:
                    print(f"    → 🏆 DENSE ADVANTAGE in {category_name}")
                else:
                    print(f"    → 🤝 EVEN SPLIT in {category_name}")

        # Query complexity analysis
        complex_queries = [r for r in results if len(r["query"].split()) >= 4]
        simple_queries = [r for r in results if len(r["query"].split()) < 4]

        print("\n🧩 COMPLEXITY ANALYSIS:")
        if complex_queries:
            complex_hybrid_wins = sum(1 for r in complex_queries if "HYBRID WINS" in r["winner"])
            print(f"  Complex Queries (4+ terms): {len(complex_queries)} total")
            print(
                f"    Hybrid wins: {complex_hybrid_wins}/{len(complex_queries)} ({complex_hybrid_wins / len(complex_queries) * 100:.1f}%)"
            )

        if simple_queries:
            simple_dense_success = sum(1 for r in simple_queries if "DENSE WINS" in r["winner"] or "TIE" in r["winner"])
            print(f"  Simple Queries (<4 terms): {len(simple_queries)} total")
            print(
                f"    Dense success: {simple_dense_success}/{len(simple_queries)} ({simple_dense_success / len(simple_queries) * 100:.1f}%)"
            )

        # Key insights and recommendations
        print("\n💡 KEY INSIGHTS:")

        # Identify hybrid's strongest categories
        hybrid_strong_categories: list[str] = []
        for category, queries in categories.items():
            if len(queries) >= 2:  # Only consider categories with multiple queries
                cat_hybrid_wins = sum(1 for r in queries if "HYBRID WINS" in r["winner"])
                cat_total = len(queries)
                if cat_hybrid_wins / cat_total > 0.6:  # >60% hybrid wins
                    hybrid_strong_categories.append(category.replace("_", " ").title())

        if hybrid_strong_categories:
            print(f"  🟡 HYBRID excels with: {', '.join(hybrid_strong_categories)}")

        # Identify dense's strongest areas
        dense_strong_categories: list[str] = []
        for category, queries in categories.items():
            if len(queries) >= 2:
                cat_dense_wins = sum(1 for r in queries if "DENSE WINS" in r["winner"])
                cat_ties = sum(1 for r in queries if "TIE" in r["winner"])
                cat_total = len(queries)
                if (cat_dense_wins + cat_ties) / cat_total > 0.6:  # >60% dense success
                    dense_strong_categories.append(category.replace("_", " ").title())

        if dense_strong_categories:
            print(f"  🔵 DENSE competitive with: {', '.join(dense_strong_categories)}")

        print("\n⚖️  RECOMMENDATIONS:")
        if hybrid_success_rate > dense_success_rate + 10:
            print("  → Use HYBRID for technical documentation and API references")
        elif dense_success_rate > hybrid_success_rate + 10:
            print("  → Use DENSE for conceptual and semantic queries")
        else:
            print("  → Performance depends on query type and content domain")

        print("  → Consider query complexity: Hybrid better for specific terms")
        print(
            f"  → Consider performance: Dense ~{abs(avg_hybrid_time - avg_dense_time) / avg_dense_time * 100:.1f}% faster on average"
        )


def main() -> None:
    """Main execution function."""
    logger.info("Starting retrieval mode comparison...")

    comparer = RetrievalComparer()

    try:
        # Setup
        comparer.setup_collections()

        # Run comparison
        comparer.run_comparison()

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise
    finally:
        # Cleanup
        try:
            comparer.client.delete_collection("dense_test")
            comparer.client.delete_collection("hybrid_test")
            logger.info("Cleaned up test collections")
        except Exception:
            pass


if __name__ == "__main__":
    main()
