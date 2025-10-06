# Lesson 5: RAG System Evaluation and Optimization

## Introduction

Production RAG systems require rigorous evaluation, continuous monitoring, and systematic optimization. This lesson teaches you to build comprehensive evaluation frameworks, implement A/B testing, optimize performance, and maintain production RAG systems at scale.

## Comprehensive Evaluation Framework

### RAG Evaluation Metrics

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
import numpy as np

@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for RAG evaluation."""
    precision: float
    recall: float
    f1_score: float
    map_score: float  # Mean Average Precision
    ndcg_score: float  # Normalized Discounted Cumulative Gain
    mrr_score: float  # Mean Reciprocal Rank
    latency_ms: float
    throughput_qps: float
    metadata: Dict = field(default_factory=dict)

class RAGEvaluator:
    """
    Comprehensive evaluation system for RAG pipelines.
    """
    def __init__(self):
        self.evaluation_history: List[Dict] = []
        self.baseline_metrics: Optional[EvaluationMetrics] = None

    def evaluate(
        self,
        rag_system: 'MultiStageRAGPipeline',
        test_queries: List[Dict],  # {'query': str, 'relevant_docs': List[str]}
        calculate_all: bool = True
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of RAG system.

        Args:
            rag_system: RAG pipeline to evaluate
            test_queries: List of queries with ground truth
            calculate_all: Whether to calculate all metrics
        """
        # Collect results
        all_results = []
        latencies = []

        for query_data in test_queries:
            start_time = datetime.now()

            results = rag_system.process_query(
                query_data['query'],
                context=query_data.get('context')
            )

            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000

            retrieved_docs = [
                doc.document.id
                for doc in results['retrieved_documents']
            ]

            all_results.append({
                'query': query_data['query'],
                'retrieved': retrieved_docs,
                'relevant': query_data['relevant_docs'],
                'latency': latency
            })
            latencies.append(latency)

        # Calculate metrics
        precision = self._calculate_precision(all_results)
        recall = self._calculate_recall(all_results)
        f1 = self._calculate_f1(precision, recall)

        map_score = 0.0
        ndcg = 0.0
        mrr = 0.0

        if calculate_all:
            map_score = self._calculate_map(all_results)
            ndcg = self._calculate_ndcg(all_results)
            mrr = self._calculate_mrr(all_results)

        # Calculate throughput
        total_time = sum(latencies) / 1000  # seconds
        throughput = len(test_queries) / total_time if total_time > 0 else 0

        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            map_score=map_score,
            ndcg_score=ndcg,
            mrr_score=mrr,
            latency_ms=np.mean(latencies),
            throughput_qps=throughput,
            metadata={
                'num_queries': len(test_queries),
                'timestamp': datetime.now()
            }
        )

        # Record evaluation
        self.evaluation_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'num_queries': len(test_queries)
        })

        return metrics

    def _calculate_precision(self, results: List[Dict]) -> float:
        """Calculate average precision across all queries."""
        precisions = []

        for result in results:
            retrieved = set(result['retrieved'])
            relevant = set(result['relevant'])

            if len(retrieved) > 0:
                precision = len(retrieved & relevant) / len(retrieved)
                precisions.append(precision)

        return float(np.mean(precisions)) if precisions else 0.0

    def _calculate_recall(self, results: List[Dict]) -> float:
        """Calculate average recall across all queries."""
        recalls = []

        for result in results:
            retrieved = set(result['retrieved'])
            relevant = set(result['relevant'])

            if len(relevant) > 0:
                recall = len(retrieved & relevant) / len(relevant)
                recalls.append(recall)

        return float(np.mean(recalls)) if recalls else 0.0

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_map(self, results: List[Dict]) -> float:
        """Calculate Mean Average Precision."""
        avg_precisions = []

        for result in results:
            retrieved = result['retrieved']
            relevant = set(result['relevant'])

            if not relevant:
                continue

            precisions_at_k = []
            num_relevant = 0

            for k, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    num_relevant += 1
                    precision_at_k = num_relevant / k
                    precisions_at_k.append(precision_at_k)

            if precisions_at_k:
                avg_precision = np.mean(precisions_at_k)
                avg_precisions.append(avg_precision)

        return float(np.mean(avg_precisions)) if avg_precisions else 0.0

    def _calculate_ndcg(self, results: List[Dict], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        ndcg_scores = []

        for result in results:
            retrieved = result['retrieved'][:k]
            relevant = set(result['relevant'])

            # Calculate DCG
            dcg = 0.0
            for i, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    dcg += 1.0 / np.log2(i + 1)

            # Calculate IDCG (perfect ranking)
            num_relevant = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

            # Calculate NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def _calculate_mrr(self, results: List[Dict]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []

        for result in results:
            retrieved = result['retrieved']
            relevant = set(result['relevant'])

            # Find rank of first relevant document
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    def set_baseline(self, metrics: EvaluationMetrics):
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics

    def compare_to_baseline(self, metrics: EvaluationMetrics) -> Dict:
        """Compare metrics to baseline."""
        if not self.baseline_metrics:
            return {}

        return {
            'precision_change': metrics.precision - self.baseline_metrics.precision,
            'recall_change': metrics.recall - self.baseline_metrics.recall,
            'f1_change': metrics.f1_score - self.baseline_metrics.f1_score,
            'latency_change': metrics.latency_ms - self.baseline_metrics.latency_ms,
            'throughput_change': metrics.throughput_qps - self.baseline_metrics.throughput_qps
        }
```

## A/B Testing Framework

### RAG System A/B Testing

```python
from typing import Callable
import random

class RAGABTester:
    """
    A/B testing framework for RAG systems.
    """
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.results: Dict[str, List[Dict]] = {}

    def create_experiment(
        self,
        experiment_id: str,
        variant_a: Callable,
        variant_b: Callable,
        traffic_split: float = 0.5,
        description: str = ""
    ):
        """
        Create new A/B test experiment.

        Args:
            experiment_id: Unique identifier
            variant_a: Control variant (function)
            variant_b: Test variant (function)
            traffic_split: Fraction of traffic to variant B
            description: Experiment description
        """
        self.experiments[experiment_id] = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'traffic_split': traffic_split,
            'description': description,
            'start_time': datetime.now(),
            'status': 'running'
        }
        self.results[experiment_id] = []

    def run_experiment(
        self,
        experiment_id: str,
        query: str,
        context: Optional[Dict] = None,
        user_id: Optional[str] = None
    ) -> Tuple[any, str]:
        """
        Run query through A/B test.

        Returns:
            (result, variant_used)
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")

        experiment = self.experiments[experiment_id]

        # Determine variant
        if user_id:
            # Consistent assignment for user
            variant = 'b' if hash(user_id) % 100 < experiment['traffic_split'] * 100 else 'a'
        else:
            # Random assignment
            variant = 'b' if random.random() < experiment['traffic_split'] else 'a'

        # Execute variant
        start_time = datetime.now()

        if variant == 'a':
            result = experiment['variant_a'](query, context)
        else:
            result = experiment['variant_b'](query, context)

        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds() * 1000

        # Record result
        self.results[experiment_id].append({
            'query': query,
            'variant': variant,
            'latency': latency,
            'timestamp': datetime.now(),
            'user_id': user_id
        })

        return result, variant

    def record_feedback(
        self,
        experiment_id: str,
        user_id: str,
        rating: float,
        relevant: bool
    ):
        """Record user feedback for experiment."""
        if experiment_id not in self.results:
            return

        # Find last result for this user
        for result in reversed(self.results[experiment_id]):
            if result.get('user_id') == user_id:
                result['rating'] = rating
                result['relevant'] = relevant
                break

    def analyze_experiment(
        self,
        experiment_id: str,
        min_samples: int = 100
    ) -> Dict:
        """
        Analyze experiment results.
        """
        if experiment_id not in self.results:
            return {}

        results = self.results[experiment_id]

        if len(results) < min_samples:
            return {
                'status': 'insufficient_data',
                'samples': len(results),
                'required': min_samples
            }

        # Split by variant
        variant_a_results = [r for r in results if r['variant'] == 'a']
        variant_b_results = [r for r in results if r['variant'] == 'b']

        # Calculate metrics for each variant
        a_metrics = self._calculate_variant_metrics(variant_a_results)
        b_metrics = self._calculate_variant_metrics(variant_b_results)

        # Statistical significance
        significance = self._test_significance(
            a_metrics,
            b_metrics,
            len(variant_a_results),
            len(variant_b_results)
        )

        return {
            'experiment_id': experiment_id,
            'status': 'complete',
            'variant_a': a_metrics,
            'variant_b': b_metrics,
            'significance': significance,
            'winner': self._determine_winner(a_metrics, b_metrics, significance)
        }

    def _calculate_variant_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for a variant."""
        if not results:
            return {}

        latencies = [r['latency'] for r in results]
        ratings = [r['rating'] for r in results if 'rating' in r]
        relevant = [r['relevant'] for r in results if 'relevant' in r]

        return {
            'count': len(results),
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'avg_rating': np.mean(ratings) if ratings else None,
            'relevance_rate': np.mean(relevant) if relevant else None
        }

    def _test_significance(
        self,
        a_metrics: Dict,
        b_metrics: Dict,
        n_a: int,
        n_b: int
    ) -> Dict:
        """Test statistical significance (simplified)."""
        # In production, use proper statistical tests (t-test, chi-square, etc.)

        if not a_metrics.get('avg_rating') or not b_metrics.get('avg_rating'):
            return {'significant': False}

        # Simple threshold-based significance
        diff = abs(b_metrics['avg_rating'] - a_metrics['avg_rating'])
        min_samples = min(n_a, n_b)

        significant = diff > 0.1 and min_samples >= 100

        return {
            'significant': significant,
            'difference': b_metrics['avg_rating'] - a_metrics['avg_rating'],
            'confidence': 0.95 if significant else 0.8
        }

    def _determine_winner(
        self,
        a_metrics: Dict,
        b_metrics: Dict,
        significance: Dict
    ) -> str:
        """Determine winning variant."""
        if not significance.get('significant'):
            return 'inconclusive'

        if significance['difference'] > 0:
            return 'variant_b'
        else:
            return 'variant_a'
```

## Performance Optimization

### RAG Performance Optimizer

```python
class RAGPerformanceOptimizer:
    """
    Optimizes RAG system performance.
    """
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.optimization_history: List[Dict] = []

    def optimize_indexing(self):
        """Optimize vector indexing for faster retrieval."""
        # Implement index optimization
        # - Rebuild indexes with better parameters
        # - Shard large indexes
        # - Use approximate nearest neighbor search

        optimizations = {
            'index_type': 'HNSW',  # Hierarchical Navigable Small World
            'ef_construction': 200,
            'ef_search': 100,
            'sharding_enabled': True
        }

        self._record_optimization('indexing', optimizations)
        return optimizations

    def optimize_caching(self):
        """Optimize caching strategy."""
        # Analyze query patterns
        # Implement multi-level cache
        # Set appropriate TTLs

        optimizations = {
            'cache_layers': ['memory', 'redis'],
            'ttl_hot': 3600,  # 1 hour
            'ttl_warm': 86400,  # 24 hours
            'max_cache_size': 10000
        }

        self._record_optimization('caching', optimizations)
        return optimizations

    def optimize_retrieval_params(self, evaluation_results: List[EvaluationMetrics]):
        """Optimize retrieval parameters based on evaluation."""
        # Analyze what works best
        best_precision = max(evaluation_results, key=lambda x: x.precision)
        best_latency = min(evaluation_results, key=lambda x: x.latency_ms)

        # Balance precision and latency
        optimizations = {
            'top_k': 15,  # Retrieve more candidates
            'rerank_top_n': 10,  # But only rerank top subset
            'similarity_threshold': 0.7,
            'use_query_expansion': True
        }

        self._record_optimization('retrieval_params', optimizations)
        return optimizations

    def optimize_batch_processing(self):
        """Enable batch processing for better throughput."""
        optimizations = {
            'batch_size': 32,
            'batch_timeout_ms': 100,
            'parallel_workers': 4
        }

        self._record_optimization('batch_processing', optimizations)
        return optimizations

    def _record_optimization(self, optimization_type: str, params: Dict):
        """Record optimization for tracking."""
        self.optimization_history.append({
            'type': optimization_type,
            'params': params,
            'timestamp': datetime.now()
        })

class ProductionRAGMonitor:
    """
    Monitors production RAG system health and performance.
    """
    def __init__(self):
        self.metrics_buffer: List[Dict] = []
        self.alerts: List[Dict] = []
        self.health_checks: List[Callable] = []

        self._register_health_checks()

    def _register_health_checks(self):
        """Register health check functions."""
        self.health_checks.extend([
            self._check_latency,
            self._check_error_rate,
            self._check_retrieval_quality,
            self._check_cache_hit_rate
        ])

    def record_query(
        self,
        query: str,
        latency: float,
        num_results: int,
        error: Optional[str] = None
    ):
        """Record query metrics."""
        self.metrics_buffer.append({
            'query': query,
            'latency': latency,
            'num_results': num_results,
            'error': error,
            'timestamp': datetime.now()
        })

    def check_health(self) -> Dict:
        """Run all health checks."""
        health_status = {
            'overall': 'healthy',
            'checks': {},
            'timestamp': datetime.now()
        }

        for check in self.health_checks:
            check_name = check.__name__
            result = check()
            health_status['checks'][check_name] = result

            if result['status'] != 'healthy':
                health_status['overall'] = 'degraded'

        return health_status

    def _check_latency(self) -> Dict:
        """Check if latency is acceptable."""
        if not self.metrics_buffer:
            return {'status': 'healthy', 'reason': 'no_data'}

        recent = self.metrics_buffer[-100:]
        latencies = [m['latency'] for m in recent]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        if p95_latency > 2000:  # 2 seconds
            return {
                'status': 'unhealthy',
                'reason': f'High P95 latency: {p95_latency:.0f}ms'
            }
        elif avg_latency > 1000:
            return {
                'status': 'degraded',
                'reason': f'Elevated average latency: {avg_latency:.0f}ms'
            }

        return {'status': 'healthy', 'avg_latency': avg_latency}

    def _check_error_rate(self) -> Dict:
        """Check error rate."""
        if not self.metrics_buffer:
            return {'status': 'healthy', 'reason': 'no_data'}

        recent = self.metrics_buffer[-100:]
        errors = sum(1 for m in recent if m.get('error'))
        error_rate = errors / len(recent)

        if error_rate > 0.05:  # 5%
            return {
                'status': 'unhealthy',
                'reason': f'High error rate: {error_rate:.1%}'
            }

        return {'status': 'healthy', 'error_rate': error_rate}

    def _check_retrieval_quality(self) -> Dict:
        """Check retrieval quality."""
        if not self.metrics_buffer:
            return {'status': 'healthy', 'reason': 'no_data'}

        recent = self.metrics_buffer[-100:]
        avg_results = np.mean([m['num_results'] for m in recent])

        if avg_results < 1:
            return {
                'status': 'unhealthy',
                'reason': 'Very low retrieval rate'
            }

        return {'status': 'healthy', 'avg_results': avg_results}

    def _check_cache_hit_rate(self) -> Dict:
        """Check cache effectiveness."""
        # Placeholder - would integrate with actual cache metrics
        return {'status': 'healthy', 'hit_rate': 0.75}
```

## Key Takeaways

1. **Comprehensive Metrics**: Evaluate precision, recall, F1, MAP, NDCG, MRR

2. **A/B Testing**: Systematic experimentation for improvements

3. **Statistical Rigor**: Ensure changes are significant

4. **Performance Optimization**: Multiple layers (indexing, caching, batching)

5. **Production Monitoring**: Continuous health checks and alerting

6. **Iterative Improvement**: Data-driven optimization cycle

## Congratulations!

You've completed Module 4! You now know how to:

- Build context-aware RAG systems
- Implement knowledge graphs with semantic indexing
- Integrate multiple knowledge sources dynamically
- Create advanced hybrid retrieval strategies
- Evaluate and optimize production RAG systems

---

**Practice Exercise**: Build a complete production RAG system with A/B testing, monitoring, and optimization. Demonstrate 90%+ precision and <200ms P95 latency on 10,000+ diverse queries.
