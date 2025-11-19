# Lesson 2: Knowledge Graphs and Semantic Indexing

## Introduction

Knowledge graphs provide structured representations of information that enable sophisticated reasoning and retrieval. Combined with semantic indexing, they create powerful foundations for context-aware AI systems that understand relationships between concepts.

## Knowledge Graph Fundamentals

### Building Knowledge Graphs

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import json

@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    type: str
    properties: Dict = field(default_factory=dict)
    embedding: Optional[any] = None

    def __hash__(self):
        return hash(self.id)

@dataclass
class Relation:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0

    def __hash__(self):
        return hash((self.source_id, self.relation_type, self.target_id))

class KnowledgeGraph:
    """
    Comprehensive knowledge graph system with advanced querying.
    """
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_index: Dict[str, Set[str]] = {}  # type -> entity_ids
        self.relation_index: Dict[str, List[Relation]] = {}  # entity_id -> relations

    def add_entity(self, entity: Entity):
        """Add entity to knowledge graph."""
        self.entities[entity.id] = entity

        # Index by type
        if entity.type not in self.entity_index:
            self.entity_index[entity.type] = set()
        self.entity_index[entity.type].add(entity.id)

        # Initialize relation index
        if entity.id not in self.relation_index:
            self.relation_index[entity.id] = []

    def add_relation(self, relation: Relation):
        """Add relation to knowledge graph."""
        # Verify entities exist
        if relation.source_id not in self.entities:
            raise ValueError(f"Source entity {relation.source_id} not found")
        if relation.target_id not in self.entities:
            raise ValueError(f"Target entity {relation.target_id} not found")

        self.relations.append(relation)

        # Index relation
        self.relation_index[relation.source_id].append(relation)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids]

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing"
    ) -> List[Tuple[Entity, Relation]]:
        """
        Get neighboring entities connected by relations.

        Args:
            entity_id: Source entity
            relation_type: Filter by relation type
            direction: 'outgoing', 'incoming', or 'both'
        """
        neighbors = []

        if direction in ["outgoing", "both"]:
            for relation in self.relation_index.get(entity_id, []):
                if relation_type and relation.relation_type != relation_type:
                    continue

                target_entity = self.entities.get(relation.target_id)
                if target_entity:
                    neighbors.append((target_entity, relation))

        if direction in ["incoming", "both"]:
            for relation in self.relations:
                if relation.target_id != entity_id:
                    continue
                if relation_type and relation.relation_type != relation_type:
                    continue

                source_entity = self.entities.get(relation.source_id)
                if source_entity:
                    neighbors.append((source_entity, relation))

        return neighbors

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[List[Tuple[Entity, Relation]]]:
        """
        Find path between two entities using BFS.
        """
        if start_id not in self.entities or end_id not in self.entities:
            return None

        # BFS to find shortest path
        queue = [(start_id, [])]
        visited = {start_id}

        while queue:
            current_id, path = queue.pop(0)

            if current_id == end_id:
                return path

            if len(path) >= max_depth:
                continue

            # Explore neighbors
            neighbors = self.get_neighbors(current_id, direction="outgoing")
            for neighbor_entity, relation in neighbors:
                if neighbor_entity.id not in visited:
                    visited.add(neighbor_entity.id)
                    new_path = path + [(neighbor_entity, relation)]
                    queue.append((neighbor_entity.id, new_path))

        return None  # No path found

    def query_subgraph(
        self,
        center_entity_id: str,
        depth: int = 2
    ) -> 'KnowledgeGraph':
        """
        Extract subgraph around an entity.
        """
        subgraph = KnowledgeGraph()

        # BFS to collect entities and relations
        queue = [(center_entity_id, 0)]
        visited = {center_entity_id}

        while queue:
            entity_id, current_depth = queue.pop(0)

            # Add entity
            entity = self.entities.get(entity_id)
            if entity:
                subgraph.add_entity(entity)

            if current_depth >= depth:
                continue

            # Add neighbors
            neighbors = self.get_neighbors(entity_id, direction="both")
            for neighbor_entity, relation in neighbors:
                if neighbor_entity.id not in visited:
                    visited.add(neighbor_entity.id)
                    queue.append((neighbor_entity.id, current_depth + 1))

                # Add relation if not already added
                if relation not in subgraph.relations:
                    try:
                        subgraph.add_relation(relation)
                    except ValueError:
                        # Entity might not be in subgraph yet
                        pass

        return subgraph

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        relation_types = {}
        for relation in self.relations:
            relation_types[relation.relation_type] = \
                relation_types.get(relation.relation_type, 0) + 1

        return {
            'num_entities': len(self.entities),
            'num_relations': len(self.relations),
            'entity_types': {
                etype: len(eids) for etype, eids in self.entity_index.items()
            },
            'relation_types': relation_types
        }
```

## Semantic Indexing System

### Vector-Based Semantic Search

```python
import numpy as np
from typing import Union

class SemanticIndex:
    """
    Semantic indexing system using embeddings for fast similarity search.
    """
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.entities: List[Entity] = []
        self.embeddings: Optional[np.ndarray] = None
        self.id_to_index: Dict[str, int] = {}

    def add_entity(self, entity: Entity, embedding: np.ndarray):
        """Add entity with embedding to index."""
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch")

        entity.embedding = embedding
        index = len(self.entities)

        self.entities.append(entity)
        self.id_to_index[entity.id] = index

        # Update embedding matrix
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_type: Optional[str] = None
    ) -> List[Tuple[Entity, float]]:
        """
        Search for similar entities using semantic similarity.
        """
        if self.embeddings is None:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices
        if filter_type:
            # Filter by type first
            filtered_indices = [
                i for i, entity in enumerate(self.entities)
                if entity.type == filter_type
            ]
            filtered_sims = similarities[filtered_indices]
            top_indices = filtered_indices[np.argsort(filtered_sims)[-top_k:][::-1]]
        else:
            top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return entities with scores
        results = []
        for idx in top_indices:
            entity = self.entities[idx]
            score = float(similarities[idx])
            results.append((entity, score))

        return results

    def search_by_entity(
        self,
        entity_id: str,
        top_k: int = 10
    ) -> List[Tuple[Entity, float]]:
        """Find entities similar to a given entity."""
        if entity_id not in self.id_to_index:
            return []

        index = self.id_to_index[entity_id]
        entity_embedding = self.embeddings[index]

        return self.search(entity_embedding, top_k=top_k + 1)[1:]  # Exclude self

class HybridGraphIndex:
    """
    Combines knowledge graph structure with semantic indexing.
    """
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.semantic_index = SemanticIndex()
        self._build_index()

    def _build_index(self):
        """Build semantic index from knowledge graph entities."""
        for entity in self.kg.entities.values():
            if entity.embedding is not None:
                self.semantic_index.add_entity(entity, entity.embedding)

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        center_entity_id: Optional[str] = None,
        top_k: int = 10,
        structure_weight: float = 0.3
    ) -> List[Tuple[Entity, float]]:
        """
        Hybrid search combining semantic similarity and graph structure.

        Args:
            query_embedding: Query vector
            center_entity_id: Optional entity to use as structural anchor
            structure_weight: Weight for structural relevance (0-1)
        """
        # Get semantic matches
        semantic_results = self.semantic_index.search(
            query_embedding,
            top_k=top_k * 3  # Get more candidates
        )

        if not center_entity_id or structure_weight == 0:
            return semantic_results[:top_k]

        # Calculate structural relevance
        structural_scores = {}
        for entity, _ in semantic_results:
            # Find path to center entity
            path = self.kg.find_path(center_entity_id, entity.id)

            if path:
                # Shorter path = higher structural relevance
                path_length = len(path)
                structural_score = 1.0 / (1.0 + path_length)
            else:
                structural_score = 0.0

            structural_scores[entity.id] = structural_score

        # Combine scores
        combined_results = []
        for entity, semantic_score in semantic_results:
            structural_score = structural_scores.get(entity.id, 0.0)

            combined_score = (
                (1 - structure_weight) * semantic_score +
                structure_weight * structural_score
            )

            combined_results.append((entity, combined_score))

        # Sort and return top-k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]

    def contextual_expansion(
        self,
        entity_ids: List[str],
        max_expansions: int = 5
    ) -> List[Entity]:
        """
        Expand entity set using graph structure and semantics.
        """
        expanded_entities = set(entity_ids)
        expansion_candidates = []

        # Collect neighbor entities
        for entity_id in entity_ids:
            neighbors = self.kg.get_neighbors(entity_id, direction="both")
            for neighbor_entity, relation in neighbors:
                if neighbor_entity.id not in expanded_entities:
                    expansion_candidates.append(
                        (neighbor_entity, relation.confidence)
                    )

        # Sort by confidence and add top candidates
        expansion_candidates.sort(key=lambda x: x[1], reverse=True)

        for entity, _ in expansion_candidates[:max_expansions]:
            expanded_entities.add(entity.id)

        return [self.kg.entities[eid] for eid in expanded_entities]
```

## Knowledge Graph Construction from Text

### Automatic Knowledge Extraction

```python
class KnowledgeExtractor:
    """
    Extract entities and relations from text to build knowledge graph.
    """
    def __init__(self):
        self.entity_patterns = {}
        self.relation_patterns = {}

    def extract_from_text(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from text.

        In production, use NER and relation extraction models.
        This is a simplified version for demonstration.
        """
        entities = self._extract_entities(text)
        relations = self._extract_relations(text, entities)

        return entities, relations

    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text (simplified)."""
        # In production, use spaCy, BERT-based NER, etc.
        entities = []

        # Simple pattern matching (replace with real NER)
        words = text.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                entity = Entity(
                    id=f"entity_{i}_{word.lower()}",
                    type="NAMED_ENTITY",
                    properties={'text': word, 'position': i}
                )
                entities.append(entity)

        return entities

    def _extract_relations(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relation]:
        """Extract relations between entities (simplified)."""
        # In production, use relation extraction models
        relations = []

        # Simple pattern: if entities appear close together, create relation
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                pos1 = entity1.properties['position']
                pos2 = entity2.properties['position']

                if abs(pos2 - pos1) <= 5:  # Within 5 words
                    relation = Relation(
                        source_id=entity1.id,
                        target_id=entity2.id,
                        relation_type="NEAR",
                        confidence=0.5
                    )
                    relations.append(relation)

        return relations

    def build_knowledge_graph(
        self,
        documents: List[str]
    ) -> KnowledgeGraph:
        """Build knowledge graph from multiple documents."""
        kg = KnowledgeGraph()

        for doc in documents:
            entities, relations = self.extract_from_text(doc)

            # Add entities
            for entity in entities:
                if entity.id not in kg.entities:
                    kg.add_entity(entity)

            # Add relations
            for relation in relations:
                try:
                    kg.add_relation(relation)
                except ValueError:
                    # Entity not in graph, skip
                    pass

        return kg
```

## Key Takeaways

1. **Knowledge Graphs**: Structured representation enables sophisticated reasoning

2. **Semantic Indexing**: Fast similarity search complements graph structure

3. **Hybrid Approaches**: Combine semantic and structural relevance

4. **Graph Traversal**: BFS/DFS for path finding and subgraph extraction

5. **Automatic Extraction**: Build graphs from unstructured text

6. **Contextual Expansion**: Use graph structure to expand query context

## What's Next

In Lesson 3, we'll explore dynamic knowledge integration and multi-source fusion.

---

**Practice Exercise**: Build a knowledge graph from 1000+ documents with semantic indexing. Implement hybrid search that outperforms pure semantic search by 15% on path-dependent queries.
