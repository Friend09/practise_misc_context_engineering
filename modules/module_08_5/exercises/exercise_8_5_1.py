"""
Exercise 8.5.1: Implementing Context Compression

Objective: Build a context compression system with multiple algorithms
Skills: Compression algorithms, information preservation, performance optimization
Duration: 120 minutes

This exercise teaches you to implement and optimize context compression algorithms
for long-running AI agents, focusing on information preservation while achieving
significant space savings.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid
import hashlib
from abc import ABC, abstractmethod
import math


@dataclass
class CompressionMetrics:
    """Metrics for evaluating compression quality"""
    original_length: int
    compressed_length: int
    compression_ratio: float
    information_preservation_score: float
    processing_time_ms: float
    memory_usage_bytes: int
    retrieval_accuracy: Optional[float] = None
    
    def __post_init__(self):
        if self.compression_ratio == 0:
            self.compression_ratio = self.compressed_length / self.original_length if self.original_length > 0 else 0


@dataclass
class ContextItem:
    """Individual context item with metadata"""
    content: str
    timestamp: datetime
    item_type: str
    importance_score: float
    metadata: Dict[str, Any]
    item_id: str = ""
    
    def __post_init__(self):
        if not self.item_id:
            self.item_id = hashlib.md5(f"{self.content}{self.timestamp}".encode()).hexdigest()


class CompressionAlgorithm(ABC):
    """Abstract base class for compression algorithms"""
    
    @abstractmethod
    def compress(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """Compress context items and return results with metadata"""
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: Dict[str, Any]) -> List[ContextItem]:
        """Decompress data back to context items"""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return algorithm name for identification"""
        pass


class HierarchicalSummarizer(CompressionAlgorithm):
    """
    Hierarchical summarization algorithm for context compression
    
    TODO: Implement hierarchical compression with multiple levels
    """
    
    def __init__(self, target_ratio: float = 5.0):
        self.target_ratio = target_ratio
        self.compression_levels = [
            {'ratio': 2, 'detail': 'high', 'preserve_keywords': True},
            {'ratio': 5, 'detail': 'medium', 'preserve_keywords': True}, 
            {'ratio': 10, 'detail': 'low', 'preserve_keywords': False}
        ]
    
    def compress(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """
        Compress context items using hierarchical summarization
        
        TODO: Implement the compression logic
        - Group context items into chunks
        - Apply progressive summarization
        - Preserve key information based on importance scores
        - Return compressed representation with metadata
        """
        # Your implementation here
        pass
    
    def decompress(self, compressed_data: Dict[str, Any]) -> List[ContextItem]:
        """
        Decompress hierarchical summaries back to context items
        
        TODO: Implement decompression logic
        - Extract summaries from compressed data
        - Reconstruct context items from summaries
        - Preserve original metadata where possible
        """
        # Your implementation here
        pass
    
    def get_algorithm_name(self) -> str:
        return "HierarchicalSummarizer"
    
    def _chunk_items(self, items: List[ContextItem], chunk_size: int) -> List[List[ContextItem]]:
        """Helper method to chunk items for processing"""
        # Your implementation here
        pass
    
    def _summarize_chunk(self, chunk: List[ContextItem], detail_level: str) -> str:
        """Helper method to summarize a chunk of items"""
        # Your implementation here
        pass


class KeyMomentExtractor(CompressionAlgorithm):
    """
    Extract and preserve key moments while compressing routine interactions
    
    TODO: Implement key moment extraction algorithm
    """
    
    def __init__(self, importance_threshold: float = 0.7):
        self.importance_threshold = importance_threshold
        self.importance_indicators = {
            'decision_made': 0.9,
            'error_occurred': 0.8,
            'goal_achieved': 0.9,
            'strategy_changed': 0.85,
            'new_information': 0.7,
            'user_feedback': 0.8,
            'breakthrough_moment': 0.95
        }
    
    def compress(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """
        Extract key moments and compress routine interactions
        
        TODO: Implement key moment extraction
        - Calculate importance scores for each item
        - Separate key moments from routine interactions
        - Create summaries for routine interactions
        - Preserve key moments with full detail
        """
        # Your implementation here
        pass
    
    def decompress(self, compressed_data: Dict[str, Any]) -> List[ContextItem]:
        """
        Reconstruct context items from key moments and summaries
        
        TODO: Implement decompression
        - Extract key moments (preserve as-is)
        - Expand routine summaries into representative items
        - Maintain temporal ordering
        """
        # Your implementation here
        pass
    
    def get_algorithm_name(self) -> str:
        return "KeyMomentExtractor"
    
    def _calculate_importance_score(self, item: ContextItem) -> float:
        """
        Calculate importance score for a context item
        
        TODO: Implement importance scoring
        - Analyze content for importance indicators
        - Consider item type and metadata
        - Apply temporal weighting
        - Return score between 0.0 and 1.0
        """
        # Your implementation here
        pass
    
    def _detect_importance_indicators(self, content: str) -> Dict[str, float]:
        """Helper method to detect importance indicators in content"""
        # Your implementation here
        pass


class SemanticClusterer(CompressionAlgorithm):
    """
    Cluster semantically similar context items for compression
    
    TODO: Implement semantic clustering algorithm
    """
    
    def __init__(self, similarity_threshold: float = 0.75, max_cluster_size: int = 10):
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size
    
    def compress(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """
        Cluster similar items and create cluster summaries
        
        TODO: Implement semantic clustering
        - Generate embeddings for context items (simulate with simple similarity)
        - Cluster items based on semantic similarity
        - Create representative summaries for each cluster
        - Preserve cluster metadata for reconstruction
        """
        # Your implementation here
        pass
    
    def decompress(self, compressed_data: Dict[str, Any]) -> List[ContextItem]:
        """
        Reconstruct items from cluster summaries
        
        TODO: Implement cluster decompression
        - Extract cluster summaries
        - Create representative items for each cluster
        - Distribute cluster metadata across items
        """
        # Your implementation here
        pass
    
    def get_algorithm_name(self) -> str:
        return "SemanticClusterer"
    
    def _calculate_similarity(self, item1: ContextItem, item2: ContextItem) -> float:
        """
        Calculate semantic similarity between two items
        
        TODO: Implement similarity calculation
        - Use simple text similarity for this exercise
        - Consider content, type, and metadata
        - Return similarity score between 0.0 and 1.0
        """
        # Your implementation here
        pass
    
    def _create_cluster_summary(self, cluster: List[ContextItem]) -> str:
        """Helper method to create summary for a cluster"""
        # Your implementation here
        pass


class TemporalCompressor(CompressionAlgorithm):
    """
    Compress context based on temporal patterns and recency
    
    TODO: Implement temporal-based compression
    """
    
    def __init__(self):
        self.time_windows = {
            'immediate': timedelta(minutes=5),
            'recent': timedelta(hours=1),
            'session': timedelta(hours=6), 
            'daily': timedelta(days=1),
            'historical': timedelta(weeks=1)
        }
        
        self.compression_strategies = {
            'immediate': 'preserve_all',
            'recent': 'selective_compression',
            'session': 'moderate_compression',
            'daily': 'aggressive_compression', 
            'historical': 'summary_only'
        }
    
    def compress(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """
        Apply temporal-based compression strategies
        
        TODO: Implement temporal compression
        - Group items by temporal windows
        - Apply appropriate compression strategy for each window
        - Preserve recent items with higher fidelity
        - Create temporal summaries for older items
        """
        # Your implementation here
        pass
    
    def decompress(self, compressed_data: Dict[str, Any]) -> List[ContextItem]:
        """
        Reconstruct items from temporal windows
        
        TODO: Implement temporal decompression
        - Extract items from each temporal window
        - Reconstruct items based on compression strategy used
        - Maintain temporal ordering
        """
        # Your implementation here
        pass
    
    def get_algorithm_name(self) -> str:
        return "TemporalCompressor"
    
    def _assign_temporal_window(self, item: ContextItem, current_time: datetime) -> str:
        """Helper method to assign item to temporal window"""
        # Your implementation here
        pass


class CompressionEvaluator:
    """
    Evaluate and compare compression algorithm performance
    
    TODO: Implement comprehensive evaluation metrics
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_compression(self, original_items: List[ContextItem], 
                           compressed_data: Dict[str, Any],
                           algorithm: CompressionAlgorithm) -> CompressionMetrics:
        """
        Evaluate compression quality across multiple metrics
        
        TODO: Implement evaluation logic
        - Calculate compression ratio
        - Assess information preservation
        - Measure processing time and memory usage
        - Evaluate retrieval accuracy
        """
        # Your implementation here
        pass
    
    def compare_algorithms(self, context_items: List[ContextItem],
                          algorithms: List[CompressionAlgorithm]) -> Dict[str, CompressionMetrics]:
        """
        Compare multiple algorithms on the same dataset
        
        TODO: Implement algorithm comparison
        - Run each algorithm on the same data
        - Collect metrics for each algorithm
        - Provide comparative analysis
        """
        # Your implementation here
        pass
    
    def calculate_information_preservation_score(self, original_items: List[ContextItem],
                                               decompressed_items: List[ContextItem]) -> float:
        """
        Calculate how well information was preserved during compression
        
        TODO: Implement preservation scoring
        - Compare original vs decompressed items
        - Account for different types of information loss
        - Return score between 0.0 and 1.0
        """
        # Your implementation here
        pass


class ContextCompressionSystem:
    """
    Main system for context compression with algorithm selection
    
    TODO: Implement adaptive compression system
    """
    
    def __init__(self):
        self.algorithms = {
            'hierarchical': HierarchicalSummarizer(),
            'key_moments': KeyMomentExtractor(),
            'semantic': SemanticClusterer(), 
            'temporal': TemporalCompressor()
        }
        self.evaluator = CompressionEvaluator()
        self.compression_history = []
    
    def compress_context(self, context_items: List[ContextItem],
                        algorithm_name: str = 'auto',
                        target_ratio: float = 5.0) -> Dict[str, Any]:
        """
        Compress context using specified or automatically selected algorithm
        
        TODO: Implement adaptive compression
        - Select best algorithm if 'auto' specified
        - Apply compression with target ratio
        - Evaluate compression quality
        - Store compression history
        """
        # Your implementation here
        pass
    
    def select_best_algorithm(self, context_items: List[ContextItem]) -> str:
        """
        Automatically select best algorithm for given context
        
        TODO: Implement algorithm selection logic
        - Analyze context characteristics
        - Consider item types and distribution
        - Select algorithm most likely to perform well
        """
        # Your implementation here
        pass
    
    def analyze_context_characteristics(self, context_items: List[ContextItem]) -> Dict[str, Any]:
        """
        Analyze context to inform algorithm selection
        
        TODO: Implement context analysis
        - Calculate temporal distribution
        - Analyze content types and importance scores
        - Identify patterns that favor specific algorithms
        """
        # Your implementation here
        pass


def create_test_context_items() -> List[ContextItem]:
    """Create test context items for algorithm testing"""
    
    items = []
    base_time = datetime.now() - timedelta(hours=2)
    
    # Create varied context items
    test_scenarios = [
        ("User asked about login issues", "user_query", 0.6),
        ("System detected authentication failure", "system_event", 0.8),
        ("Applied password reset procedure", "action", 0.9),
        ("User confirmed email received", "user_feedback", 0.7),
        ("Password reset completed successfully", "outcome", 0.9),
        ("User asked about account features", "user_query", 0.4),
        ("Provided feature overview", "response", 0.5),
        ("User requested advanced tutorials", "user_query", 0.6),
        ("Shared tutorial links", "response", 0.4),
        ("User reported tutorial was helpful", "user_feedback", 0.8)
    ]
    
    for i, (content, item_type, importance) in enumerate(test_scenarios):
        items.append(ContextItem(
            content=content,
            timestamp=base_time + timedelta(minutes=i*10),
            item_type=item_type,
            importance_score=importance,
            metadata={'test_item': True, 'sequence': i}
        ))
    
    return items


def test_compression_algorithms():
    """Test compression algorithm implementations"""
    print("Testing Context Compression Algorithms")
    print("=" * 50)
    
    # Create test data
    test_items = create_test_context_items()
    print(f"Created {len(test_items)} test context items")
    
    # Initialize system
    compression_system = ContextCompressionSystem()
    
    # Test each algorithm
    algorithms_to_test = ['hierarchical', 'key_moments', 'semantic', 'temporal']
    
    results = {}
    for alg_name in algorithms_to_test:
        print(f"\nTesting {alg_name} algorithm...")
        try:
            result = compression_system.compress_context(test_items, alg_name)
            results[alg_name] = result
            print(f"  Status: {'SUCCESS' if result else 'FAILED'}")
        except NotImplementedError:
            print(f"  Status: NOT IMPLEMENTED")
        except Exception as e:
            print(f"  Status: ERROR - {str(e)}")
    
    # Test algorithm comparison
    print(f"\nTesting algorithm comparison...")
    try:
        comparison = compression_system.evaluator.compare_algorithms(
            test_items, 
            list(compression_system.algorithms.values())
        )
        print(f"  Comparison results: {'SUCCESS' if comparison else 'FAILED'}")
    except NotImplementedError:
        print(f"  Comparison: NOT IMPLEMENTED")
    
    print("\nExercise 8.5.1 Implementation Status:")
    print("- Implement all TODO methods in the classes above")
    print("- All algorithms should compress and decompress successfully")
    print("- Evaluator should provide meaningful metrics")
    print("- System should select appropriate algorithms automatically")


def main():
    """Main function to run the exercise"""
    print("Context Engineering Exercise 8.5.1: Context Compression")
    print("=" * 60)
    print()
    
    print("This exercise focuses on:")
    print("1. Implementing hierarchical summarization")
    print("2. Building key moment extraction systems")
    print("3. Creating semantic clustering algorithms")
    print("4. Developing temporal compression strategies")
    print("5. Evaluating compression quality and performance")
    print()
    
    print("Your tasks:")
    print("1. Implement all compression algorithms")
    print("2. Create comprehensive evaluation metrics")
    print("3. Build adaptive algorithm selection")
    print("4. Optimize for information preservation")
    print("5. Test with realistic context scenarios")
    print()
    
    # Run tests
    test_compression_algorithms()
    
    print("\nNext Steps:")
    print("- Complete all TODO implementations")
    print("- Experiment with different compression parameters")
    print("- Test on longer context sequences")
    print("- Move to Exercise 8.5.2 when ready")


if __name__ == "__main__":
    main()