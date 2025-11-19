# Lesson 1: Context-Aware RAG Architecture

## Introduction

Context-Aware Retrieval-Augmented Generation (RAG) systems revolutionize how AI agents access and utilize external knowledge. By integrating context engineering principles with retrieval mechanisms, we create systems that understand not just what to retrieve, but why and how to integrate that knowledge effectively.

## RAG System Architecture Fundamentals

### Core Components

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import numpy as np

@dataclass
class Document:
    """Represents a document in the knowledge base."""
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)

@dataclass
class RetrievalContext:
    """Context information for retrieval."""
    query: str
    user_profile: Optional[Dict] = None
    conversation_history: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    constraints: Dict = field(default_factory=dict)
    preferences: Dict = field(default_factory=dict)

@dataclass
class RetrievedDocument:
    """Document with retrieval metadata."""
    document: Document
    score: float
    retrieval_method: str
    rank: int
    explanation: Optional[str] = None

class ContextAwareRetriever:
    """
    Advanced retriever that integrates context into retrieval decisions.
    """
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        top_k: int = 10
    ):
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.documents: Dict[str, Document] = {}
        self.retrieval_history: List[Dict] = []

    def add_documents(self, documents: List[Document]):
        """Add documents to the knowledge base."""
        for doc in documents:
            if doc.embedding is None and self.embedding_model:
                doc.embedding = self._embed_text(doc.content)
            self.documents[doc.id] = doc

    def retrieve(
        self,
        retrieval_context: RetrievalContext
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents with context awareness.
        """
        # Stage 1: Initial semantic retrieval
        semantic_results = self._semantic_retrieval(
            retrieval_context.query
        )

        # Stage 2: Context filtering
        filtered_results = self._apply_context_filters(
            semantic_results,
            retrieval_context
        )

        # Stage 3: Contextual re-ranking
        ranked_results = self._contextual_rerank(
            filtered_results,
            retrieval_context
        )

        # Record retrieval for analysis
        self._record_retrieval(retrieval_context, ranked_results)

        return ranked_results[:self.top_k]

    def _semantic_retrieval(self, query: str) -> List[RetrievedDocument]:
        """Perform semantic similarity search."""
        if not self.embedding_model:
            # Fallback to keyword matching
            return self._keyword_retrieval(query)

        query_embedding = self._embed_text(query)

        # Calculate similarity scores
        scored_docs = []
        for doc_id, doc in self.documents.items():
            if doc.embedding is not None:
                similarity = self._cosine_similarity(
                    query_embedding,
                    doc.embedding
                )
                scored_docs.append(
                    RetrievedDocument(
                        document=doc,
                        score=similarity,
                        retrieval_method="semantic",
                        rank=0
                    )
                )

        # Sort by score
        scored_docs.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, doc in enumerate(scored_docs):
            doc.rank = i + 1

        return scored_docs

    def _apply_context_filters(
        self,
        results: List[RetrievedDocument],
        context: RetrievalContext
    ) -> List[RetrievedDocument]:
        """Apply context-based filtering."""
        filtered = []

        for retrieved_doc in results:
            doc = retrieved_doc.document

            # Domain filtering
            if context.domain:
                doc_domain = doc.metadata.get('domain')
                if doc_domain and doc_domain != context.domain:
                    continue

            # Constraint filtering
            if context.constraints:
                # Check recency constraint
                if 'max_age_days' in context.constraints:
                    max_age = context.constraints['max_age_days']
                    age_days = (datetime.now() - doc.timestamp).days
                    if age_days > max_age:
                        continue

                # Check source constraint
                if 'allowed_sources' in context.constraints:
                    doc_source = doc.metadata.get('source')
                    if doc_source not in context.constraints['allowed_sources']:
                        continue

            filtered.append(retrieved_doc)

        return filtered

    def _contextual_rerank(
        self,
        results: List[RetrievedDocument],
        context: RetrievalContext
    ) -> List[RetrievedDocument]:
        """Re-rank results based on context."""
        for retrieved_doc in results:
            # Start with semantic score
            contextual_score = retrieved_doc.score

            # Boost based on user preferences
            if context.preferences:
                preference_boost = self._calculate_preference_boost(
                    retrieved_doc.document,
                    context.preferences
                )
                contextual_score *= (1.0 + preference_boost)

            # Boost based on conversation history relevance
            if context.conversation_history:
                history_relevance = self._calculate_history_relevance(
                    retrieved_doc.document,
                    context.conversation_history
                )
                contextual_score *= (1.0 + history_relevance * 0.2)

            # Update score
            retrieved_doc.score = contextual_score

        # Re-sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, doc in enumerate(results):
            doc.rank = i + 1

        return results

    def _calculate_preference_boost(
        self,
        document: Document,
        preferences: Dict
    ) -> float:
        """Calculate boost based on user preferences."""
        boost = 0.0

        # Preferred document types
        if 'document_types' in preferences:
            doc_type = document.metadata.get('type')
            if doc_type in preferences['document_types']:
                boost += 0.3

        # Preferred sources
        if 'sources' in preferences:
            doc_source = document.metadata.get('source')
            if doc_source in preferences['sources']:
                boost += 0.2

        return boost

    def _calculate_history_relevance(
        self,
        document: Document,
        history: List[str]
    ) -> float:
        """Calculate relevance to conversation history."""
        if not history:
            return 0.0

        # Simple keyword overlap (in production, use embeddings)
        doc_words = set(document.content.lower().split())
        history_words = set(' '.join(history).lower().split())

        overlap = len(doc_words & history_words)
        relevance = min(overlap / 100.0, 1.0)

        return relevance

    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Placeholder - integrate with actual embedding model
        return np.random.rand(384)

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def _keyword_retrieval(self, query: str) -> List[RetrievedDocument]:
        """Fallback keyword-based retrieval."""
        query_words = set(query.lower().split())
        scored_docs = []

        for doc_id, doc in self.documents.items():
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)

            if overlap > 0:
                score = overlap / len(query_words)
                scored_docs.append(
                    RetrievedDocument(
                        document=doc,
                        score=score,
                        retrieval_method="keyword",
                        rank=0
                    )
                )

        scored_docs.sort(key=lambda x: x.score, reverse=True)
        for i, doc in enumerate(scored_docs):
            doc.rank = i + 1

        return scored_docs

    def _record_retrieval(
        self,
        context: RetrievalContext,
        results: List[RetrievedDocument]
    ):
        """Record retrieval for analysis."""
        self.retrieval_history.append({
            'timestamp': datetime.now(),
            'query': context.query,
            'num_results': len(results),
            'top_score': results[0].score if results else 0.0,
            'domain': context.domain
        })
```

## Multi-Stage Retrieval Pipeline

### Advanced Pipeline with Context Integration

```python
class MultiStageRAGPipeline:
    """
    Complete RAG pipeline with multiple retrieval and processing stages.
    """
    def __init__(self):
        self.retriever = ContextAwareRetriever()
        self.reranker = ContextualReranker()
        self.integrator = KnowledgeIntegrator()
        self.cache = RetrievalCache()

    def process_query(
        self,
        query: str,
        context: Optional[RetrievalContext] = None
    ) -> Dict:
        """
        Process query through complete RAG pipeline.
        """
        if context is None:
            context = RetrievalContext(query=query)

        # Check cache first
        cached_result = self.cache.get(query, context)
        if cached_result:
            return cached_result

        # Stage 1: Retrieval
        retrieved_docs = self.retriever.retrieve(context)

        # Stage 2: Re-ranking
        reranked_docs = self.reranker.rerank(retrieved_docs, context)

        # Stage 3: Knowledge integration
        integrated_knowledge = self.integrator.integrate(
            reranked_docs,
            context
        )

        result = {
            'retrieved_documents': reranked_docs,
            'integrated_knowledge': integrated_knowledge,
            'metadata': {
                'num_documents': len(reranked_docs),
                'retrieval_time': datetime.now(),
                'query': query
            }
        }

        # Cache result
        self.cache.store(query, context, result)

        return result

class ContextualReranker:
    """
    Advanced re-ranking with multiple signals.
    """
    def __init__(self):
        self.reranking_models: Dict[str, Callable] = {}
        self._register_default_models()

    def _register_default_models(self):
        """Register default re-ranking models."""
        self.reranking_models['relevance'] = self._relevance_score
        self.reranking_models['freshness'] = self._freshness_score
        self.reranking_models['authority'] = self._authority_score
        self.reranking_models['diversity'] = self._diversity_score

    def rerank(
        self,
        documents: List[RetrievedDocument],
        context: RetrievalContext,
        weights: Optional[Dict[str, float]] = None
    ) -> List[RetrievedDocument]:
        """
        Re-rank documents using multiple signals.
        """
        if weights is None:
            weights = {
                'relevance': 0.4,
                'freshness': 0.2,
                'authority': 0.2,
                'diversity': 0.2
            }

        # Calculate scores for each model
        for doc in documents:
            combined_score = 0.0

            for model_name, weight in weights.items():
                if model_name in self.reranking_models:
                    model_score = self.reranking_models[model_name](
                        doc,
                        context
                    )
                    combined_score += model_score * weight

            doc.score = combined_score

        # Sort by new scores
        documents.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, doc in enumerate(documents):
            doc.rank = i + 1

        return documents

    def _relevance_score(
        self,
        doc: RetrievedDocument,
        context: RetrievalContext
    ) -> float:
        """Calculate relevance score."""
        # Use original retrieval score as base
        return doc.score

    def _freshness_score(
        self,
        doc: RetrievedDocument,
        context: RetrievalContext
    ) -> float:
        """Calculate freshness score."""
        age_days = (datetime.now() - doc.document.timestamp).days
        # Exponential decay
        freshness = np.exp(-age_days / 30.0)
        return float(freshness)

    def _authority_score(
        self,
        doc: RetrievedDocument,
        context: RetrievalContext
    ) -> float:
        """Calculate authority score based on source."""
        authority_scores = {
            'official': 1.0,
            'verified': 0.8,
            'community': 0.6,
            'unknown': 0.4
        }

        source_type = doc.document.metadata.get('source_type', 'unknown')
        return authority_scores.get(source_type, 0.4)

    def _diversity_score(
        self,
        doc: RetrievedDocument,
        context: RetrievalContext
    ) -> float:
        """Calculate diversity contribution score."""
        # In production, compare with already selected documents
        # For now, use a simple heuristic
        return 0.5 + np.random.rand() * 0.5

class KnowledgeIntegrator:
    """
    Integrates retrieved knowledge into coherent response.
    """
    def integrate(
        self,
        documents: List[RetrievedDocument],
        context: RetrievalContext
    ) -> Dict:
        """
        Integrate knowledge from multiple documents.
        """
        # Extract key information
        key_points = self._extract_key_points(documents)

        # Resolve contradictions
        resolved_info = self._resolve_contradictions(key_points)

        # Synthesize coherent knowledge
        synthesis = self._synthesize(resolved_info, context)

        return {
            'synthesis': synthesis,
            'sources': [doc.document.id for doc in documents],
            'confidence': self._calculate_confidence(documents)
        }

    def _extract_key_points(
        self,
        documents: List[RetrievedDocument]
    ) -> List[Dict]:
        """Extract key points from documents."""
        key_points = []

        for doc in documents:
            # In production, use extraction model
            # For now, use simple heuristic
            key_points.append({
                'content': doc.document.content[:200],
                'source': doc.document.id,
                'score': doc.score
            })

        return key_points

    def _resolve_contradictions(self, key_points: List[Dict]) -> List[Dict]:
        """Resolve contradictory information."""
        # In production, use NLI models
        # For now, return all points
        return key_points

    def _synthesize(
        self,
        information: List[Dict],
        context: RetrievalContext
    ) -> str:
        """Synthesize information into coherent knowledge."""
        # In production, use generation model
        synthesis = f"Based on {len(information)} sources: "
        synthesis += " ".join([info['content'] for info in information[:3]])
        return synthesis

    def _calculate_confidence(
        self,
        documents: List[RetrievedDocument]
    ) -> float:
        """Calculate confidence in integrated knowledge."""
        if not documents:
            return 0.0

        # Average of top document scores
        top_scores = [doc.score for doc in documents[:3]]
        return float(np.mean(top_scores))

class RetrievalCache:
    """
    Cache for retrieval results.
    """
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}

    def _generate_key(
        self,
        query: str,
        context: RetrievalContext
    ) -> str:
        """Generate cache key."""
        # Simple key generation
        return f"{query}:{context.domain}:{hash(str(context.constraints))}"

    def get(
        self,
        query: str,
        context: RetrievalContext
    ) -> Optional[Dict]:
        """Get cached result."""
        key = self._generate_key(query, context)

        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]

        return None

    def store(
        self,
        query: str,
        context: RetrievalContext,
        result: Dict
    ):
        """Store result in cache."""
        key = self._generate_key(query, context)

        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = result
        self.access_times[key] = datetime.now()

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
```

## Key Takeaways

1. **Context Integration**: Retrieval decisions should consider user context, conversation history, and preferences

2. **Multi-Stage Processing**: Use retrieval, filtering, re-ranking, and integration stages

3. **Flexible Reranking**: Combine multiple signals (relevance, freshness, authority, diversity)

4. **Knowledge Integration**: Synthesize information from multiple sources coherently

5. **Performance Optimization**: Use caching and efficient data structures

6. **Contextual Awareness**: Every stage should leverage available context

## What's Next

In Lesson 2, we'll explore knowledge graphs and semantic indexing for more sophisticated retrieval strategies.

---

**Practice Exercise**: Build a complete context-aware RAG system. Test with diverse queries and verify that context improves relevance by at least 25% over baseline semantic search.
