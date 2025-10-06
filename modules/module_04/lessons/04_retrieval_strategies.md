# Lesson 4: Advanced Retrieval Strategies

## Introduction

Advanced retrieval strategies combine multiple search methods—semantic, keyword, structural, and contextual—to achieve superior results. This lesson teaches you to build hybrid retrieval systems that leverage the strengths of different approaches while minimizing their weaknesses.

## Hybrid Retrieval Architecture

### Multi-Method Retrieval System

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
import numpy as np

class RetrievalMethod(Enum):
    """Available retrieval methods."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    STRUCTURED = "structured"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"

@dataclass
class SearchResult:
    """Result from a search method."""
    document_id: str
    score: float
    method: RetrievalMethod
    explanation: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class HybridRetrievalEngine:
    """
    Advanced retrieval engine combining multiple search methods.
    """
    def __init__(self):
        self.retrieval_methods: Dict[RetrievalMethod, Callable] = {}
        self.method_weights: Dict[RetrievalMethod, float] = {
            RetrievalMethod.SEMANTIC: 0.4,
            RetrievalMethod.KEYWORD: 0.3,
            RetrievalMethod.CONTEXTUAL: 0.3
        }
        self.performance_tracker = PerformanceTracker()

        self._register_methods()

    def _register_methods(self):
        """Register available retrieval methods."""
        self.retrieval_methods[RetrievalMethod.SEMANTIC] = self._semantic_search
        self.retrieval_methods[RetrievalMethod.KEYWORD] = self._keyword_search
        self.retrieval_methods[RetrievalMethod.CONTEXTUAL] = self._contextual_search

    def search(
        self,
        query: str,
        context: Optional[Dict] = None,
        methods: Optional[List[RetrievalMethod]] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Execute hybrid search using multiple methods.
        """
        if methods is None:
            methods = [
                RetrievalMethod.SEMANTIC,
                RetrievalMethod.KEYWORD,
                RetrievalMethod.CONTEXTUAL
            ]

        # Execute each method
        method_results = {}
        for method in methods:
            if method in self.retrieval_methods:
                results = self.retrieval_methods[method](query, context)
                method_results[method] = results

        # Fuse results
        fused_results = self._fuse_results(method_results, top_k)

        # Track performance
        self.performance_tracker.record_search(query, methods, fused_results)

        return fused_results

    def _semantic_search(
        self,
        query: str,
        context: Optional[Dict]
    ) -> List[SearchResult]:
        """Semantic similarity search."""
        # In production, use vector database
        # Placeholder implementation
        results = []

        # Simulate semantic search
        for i in range(20):
            score = 0.9 - (i * 0.03)
            results.append(SearchResult(
                document_id=f"doc_sem_{i}",
                score=score,
                method=RetrievalMethod.SEMANTIC,
                explanation="High semantic similarity"
            ))

        return results

    def _keyword_search(
        self,
        query: str,
        context: Optional[Dict]
    ) -> List[SearchResult]:
        """Keyword-based search."""
        # BM25 or similar algorithm
        results = []

        query_terms = query.lower().split()

        # Simulate keyword matching
        for i in range(15):
            score = 0.85 - (i * 0.04)
            results.append(SearchResult(
                document_id=f"doc_kw_{i}",
                score=score,
                method=RetrievalMethod.KEYWORD,
                explanation=f"Matches keywords: {', '.join(query_terms[:2])}"
            ))

        return results

    def _contextual_search(
        self,
        query: str,
        context: Optional[Dict]
    ) -> List[SearchResult]:
        """Context-aware search."""
        results = []

        # Use context to boost relevant documents
        context_terms = []
        if context:
            context_terms = context.get('relevant_terms', [])

        # Simulate contextual search
        for i in range(12):
            score = 0.80 - (i * 0.05)
            results.append(SearchResult(
                document_id=f"doc_ctx_{i}",
                score=score,
                method=RetrievalMethod.CONTEXTUAL,
                explanation=f"Relevant to context: {context_terms[:2]}"
            ))

        return results

    def _fuse_results(
        self,
        method_results: Dict[RetrievalMethod, List[SearchResult]],
        top_k: int
    ) -> List[SearchResult]:
        """
        Fuse results from multiple methods.
        """
        # Collect all unique documents
        doc_scores: Dict[str, Dict] = {}

        for method, results in method_results.items():
            weight = self.method_weights.get(method, 0.3)

            for result in results:
                doc_id = result.document_id

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'scores': {},
                        'methods': set(),
                        'explanations': []
                    }

                doc_scores[doc_id]['scores'][method] = result.score * weight
                doc_scores[doc_id]['methods'].add(method)
                doc_scores[doc_id]['explanations'].append(result.explanation)

        # Calculate combined scores
        fused_results = []
        for doc_id, info in doc_scores.items():
            # Reciprocal Rank Fusion + weighted scores
            combined_score = sum(info['scores'].values())

            # Boost documents that appear in multiple methods
            method_diversity_boost = len(info['methods']) * 0.1
            combined_score += method_diversity_boost

            fused_results.append(SearchResult(
                document_id=doc_id,
                score=combined_score,
                method=RetrievalMethod.HYBRID,
                explanation=f"Combined from {len(info['methods'])} methods",
                metadata={
                    'methods': list(info['methods']),
                    'individual_scores': info['scores']
                }
            ))

        # Sort and return top-k
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:top_k]

    def update_weights(self, feedback: Dict[RetrievalMethod, float]):
        """Update method weights based on feedback."""
        # Normalize weights
        total = sum(feedback.values())
        if total > 0:
            self.method_weights = {
                method: weight / total
                for method, weight in feedback.items()
            }

class PerformanceTracker:
    """Track retrieval performance for optimization."""
    def __init__(self):
        self.search_history: List[Dict] = []
        self.method_performance: Dict[RetrievalMethod, List[float]] = {}

    def record_search(
        self,
        query: str,
        methods: List[RetrievalMethod],
        results: List[SearchResult]
    ):
        """Record search for analysis."""
        self.search_history.append({
            'query': query,
            'methods': methods,
            'num_results': len(results),
            'top_score': results[0].score if results else 0.0,
            'timestamp': datetime.now()
        })

    def get_method_stats(self) -> Dict:
        """Get performance statistics by method."""
        stats = {}

        for method in RetrievalMethod:
            if method in self.method_performance:
                scores = self.method_performance[method]
                stats[method.value] = {
                    'avg_score': np.mean(scores) if scores else 0,
                    'count': len(scores)
                }

        return stats
```

## Contextual Query Expansion

### Intelligent Query Enhancement

```python
class QueryExpansionEngine:
    """
    Expands queries with contextually relevant terms.
    """
    def __init__(self):
        self.expansion_cache: Dict[str, List[str]] = {}
        self.term_cooccurrence: Dict[str, Dict[str, int]] = {}

    def expand_query(
        self,
        query: str,
        context: Optional[Dict] = None,
        max_terms: int = 5
    ) -> str:
        """
        Expand query with related terms.
        """
        original_terms = query.lower().split()
        expansion_terms = set(original_terms)

        # Add synonyms
        synonyms = self._get_synonyms(original_terms)
        expansion_terms.update(synonyms[:max_terms])

        # Add contextual terms
        if context:
            contextual_terms = self._get_contextual_terms(
                original_terms,
                context
            )
            expansion_terms.update(contextual_terms[:max_terms])

        # Add co-occurring terms
        cooccurring = self._get_cooccurring_terms(original_terms)
        expansion_terms.update(cooccurring[:max_terms])

        return ' '.join(expansion_terms)

    def _get_synonyms(self, terms: List[str]) -> List[str]:
        """Get synonyms for terms."""
        # In production, use WordNet or embedding-based synonyms
        synonyms = []

        synonym_map = {
            'quick': ['fast', 'rapid', 'swift'],
            'good': ['excellent', 'great', 'quality'],
            'bad': ['poor', 'low-quality', 'inferior']
        }

        for term in terms:
            if term in synonym_map:
                synonyms.extend(synonym_map[term])

        return synonyms

    def _get_contextual_terms(
        self,
        terms: List[str],
        context: Dict
    ) -> List[str]:
        """Get terms relevant to context."""
        contextual = []

        # Extract from context
        if 'domain' in context:
            domain = context['domain']
            # Add domain-specific terms
            domain_terms = {
                'medical': ['diagnosis', 'treatment', 'patient'],
                'legal': ['contract', 'liability', 'court'],
                'technical': ['system', 'implementation', 'architecture']
            }
            contextual.extend(domain_terms.get(domain, []))

        if 'user_interests' in context:
            contextual.extend(context['user_interests'])

        return contextual

    def _get_cooccurring_terms(self, terms: List[str]) -> List[str]:
        """Get terms that frequently co-occur."""
        cooccurring = []

        for term in terms:
            if term in self.term_cooccurrence:
                # Get top co-occurring terms
                term_counts = self.term_cooccurrence[term]
                sorted_terms = sorted(
                    term_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                cooccurring.extend([t for t, _ in sorted_terms[:3]])

        return cooccurring

    def learn_cooccurrence(self, documents: List[str]):
        """Learn term co-occurrences from documents."""
        for doc in documents:
            terms = doc.lower().split()

            # Update co-occurrence counts
            for i, term1 in enumerate(terms):
                if term1 not in self.term_cooccurrence:
                    self.term_cooccurrence[term1] = {}

                # Count co-occurrences in window
                window_size = 5
                for j in range(max(0, i - window_size), min(len(terms), i + window_size + 1)):
                    if i != j:
                        term2 = terms[j]
                        self.term_cooccurrence[term1][term2] = \
                            self.term_cooccurrence[term1].get(term2, 0) + 1

class QueryRewriter:
    """
    Rewrites queries for better retrieval.
    """
    def __init__(self):
        self.rewrite_patterns: List[Tuple[str, str]] = []
        self._load_patterns()

    def _load_patterns(self):
        """Load rewrite patterns."""
        # Common query patterns to rewrite
        self.rewrite_patterns = [
            ('how to', 'steps for'),
            ('what is', 'definition of'),
            ('best way', 'optimal approach'),
        ]

    def rewrite(self, query: str) -> List[str]:
        """
        Generate alternative query formulations.
        """
        rewrites = [query]

        # Apply patterns
        for pattern, replacement in self.rewrite_patterns:
            if pattern in query.lower():
                rewritten = query.lower().replace(pattern, replacement)
                rewrites.append(rewritten)

        # Generate variations
        rewrites.extend(self._generate_variations(query))

        return rewrites[:5]  # Return top 5 variations

    def _generate_variations(self, query: str) -> List[str]:
        """Generate query variations."""
        variations = []

        # More specific
        variations.append(f"{query} details")

        # More general
        words = query.split()
        if len(words) > 2:
            variations.append(' '.join(words[:-1]))

        return variations
```

## Learning-Based Retrieval Optimization

### Adaptive Retrieval System

```python
class AdaptiveRetrievalSystem:
    """
    Learns from feedback to optimize retrieval.
    """
    def __init__(self, base_retriever: HybridRetrievalEngine):
        self.retriever = base_retriever
        self.feedback_history: List[Dict] = []
        self.query_strategy_map: Dict[str, Dict] = {}

    def search_with_learning(
        self,
        query: str,
        context: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search with learned optimizations.
        """
        # Predict best strategy for this query type
        strategy = self._predict_strategy(query, context)

        # Execute search with optimized parameters
        results = self.retriever.search(
            query,
            context,
            methods=strategy['methods'],
            top_k=top_k
        )

        return results

    def record_feedback(
        self,
        query: str,
        results: List[SearchResult],
        relevant_docs: List[str],
        context: Optional[Dict] = None
    ):
        """
        Record user feedback for learning.
        """
        # Calculate metrics
        precision = self._calculate_precision(results, relevant_docs)
        recall = self._calculate_recall(results, relevant_docs)

        feedback = {
            'query': query,
            'precision': precision,
            'recall': recall,
            'methods_used': [r.method for r in results],
            'context': context,
            'timestamp': datetime.now()
        }

        self.feedback_history.append(feedback)

        # Update strategy mapping
        self._update_strategy_map(query, feedback)

    def _predict_strategy(
        self,
        query: str,
        context: Optional[Dict]
    ) -> Dict:
        """
        Predict best retrieval strategy for query.
        """
        # Classify query type
        query_type = self._classify_query(query)

        # Get historical performance for this type
        if query_type in self.query_strategy_map:
            strategy = self.query_strategy_map[query_type]
        else:
            # Default strategy
            strategy = {
                'methods': [
                    RetrievalMethod.SEMANTIC,
                    RetrievalMethod.KEYWORD,
                    RetrievalMethod.CONTEXTUAL
                ],
                'weights': self.retriever.method_weights
            }

        return strategy

    def _classify_query(self, query: str) -> str:
        """Classify query into type."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['how', 'steps', 'guide']):
            return 'procedural'
        elif any(word in query_lower for word in ['what', 'define', 'explain']):
            return 'definitional'
        elif any(word in query_lower for word in ['best', 'compare', 'vs']):
            return 'comparative'
        else:
            return 'general'

    def _update_strategy_map(self, query: str, feedback: Dict):
        """Update strategy mapping based on feedback."""
        query_type = self._classify_query(query)

        if query_type not in self.query_strategy_map:
            self.query_strategy_map[query_type] = {
                'methods': feedback['methods_used'],
                'avg_precision': feedback['precision'],
                'count': 1
            }
        else:
            # Update running average
            strategy = self.query_strategy_map[query_type]
            count = strategy['count']

            strategy['avg_precision'] = (
                strategy['avg_precision'] * count + feedback['precision']
            ) / (count + 1)
            strategy['count'] += 1

    def _calculate_precision(
        self,
        results: List[SearchResult],
        relevant_docs: List[str]
    ) -> float:
        """Calculate precision@k."""
        if not results:
            return 0.0

        relevant_retrieved = sum(
            1 for r in results if r.document_id in relevant_docs
        )

        return relevant_retrieved / len(results)

    def _calculate_recall(
        self,
        results: List[SearchResult],
        relevant_docs: List[str]
    ) -> float:
        """Calculate recall."""
        if not relevant_docs:
            return 0.0

        relevant_retrieved = sum(
            1 for r in results if r.document_id in relevant_docs
        )

        return relevant_retrieved / len(relevant_docs)
```

## Key Takeaways

1. **Hybrid Methods**: Combine semantic, keyword, and contextual search

2. **Result Fusion**: Intelligent aggregation of multiple retrieval methods

3. **Query Expansion**: Enhance queries with synonyms and context

4. **Query Rewriting**: Generate alternative formulations

5. **Learning Systems**: Adapt strategies based on feedback

6. **Performance Tracking**: Monitor and optimize retrieval quality

## What's Next

In Lesson 5, we'll explore RAG system evaluation and production optimization.

---

**Practice Exercise**: Build a hybrid retrieval system that combines 3+ methods. Show that it outperforms any single method by 20%+ on diverse query types using learned optimizations.
