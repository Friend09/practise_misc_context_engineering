# Lesson 1: Context Compression Fundamentals

## The Context Compression Challenge

As AI agents handle longer conversations and more complex tasks, naive context management quickly becomes a bottleneck. Modern language models have context windows ranging from 4K to 128K tokens, but real-world applications often require much more:

- **Extended conversations** spanning hours or days
- **Complex multi-step tasks** with rich decision history  
- **Accumulated knowledge** from multiple interactions
- **Tool execution results** and state changes
- **User preferences** and behavioral patterns learned over time

**The Core Problem**: How do we preserve essential information while staying within context limits and maintaining high performance?

## Context Compression Theory

### Information Density Principle

Context compression is fundamentally about maximizing **information density** - the amount of useful information per token consumed in the context window.

```
Information Density = Useful Information / Token Count
```

**Key Insight**: Not all tokens are created equal. Some tokens carry critical decision-making information, while others are redundant or low-value.

### Compression Quality Metrics

#### 1. Compression Ratio
```
Compression Ratio = Compressed Size / Original Size
```
- Target: 5:1 to 10:1 compression for most applications
- Higher ratios possible for routine interactions

#### 2. Information Preservation Score
```
Preservation Score = Critical Information Retained / Total Critical Information
```
- Target: >90% for high-stakes applications
- >70% acceptable for routine tasks

#### 3. Retrieval Accuracy
```
Retrieval Accuracy = Correctly Retrieved Context / Total Retrieval Attempts
```
- Measures ability to find relevant information post-compression
- Target: >95% for production systems

#### 4. Performance Impact
```
Performance Impact = (Uncompressed Performance - Compressed Performance) / Uncompressed Performance
```
- Target: <10% performance degradation
- <5% for critical applications

## Core Compression Algorithms

### 1. Hierarchical Summarization

**Concept**: Create multi-level summaries with different levels of detail.

```python
class HierarchicalSummarizer:
    """Multi-level context summarization with information preservation"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.compression_levels = [
            {'ratio': 2, 'detail': 'high', 'preserve': 'critical_decisions'},
            {'ratio': 5, 'detail': 'medium', 'preserve': 'key_outcomes'},  
            {'ratio': 10, 'detail': 'low', 'preserve': 'essential_facts'}
        ]
    
    def compress_hierarchically(self, context: str, target_ratio: float) -> Dict:
        """Apply hierarchical compression to achieve target ratio"""
        
        # Select appropriate compression level
        level = self.select_compression_level(target_ratio)
        
        # Chunk context into manageable pieces
        chunks = self.chunk_context(context, level['chunk_size'])
        
        # First pass: Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarize_chunk(chunk, level)
            chunk_summaries.append(summary)
        
        # Second pass: Combine and compress summaries
        combined_summary = self.combine_summaries(chunk_summaries, level)
        
        # Third pass: Apply final compression if needed
        if len(combined_summary) > self.calculate_target_length(context, target_ratio):
            combined_summary = self.apply_final_compression(combined_summary, level)
        
        return {
            'compressed_context': combined_summary,
            'original_length': len(context),
            'compressed_length': len(combined_summary),
            'compression_ratio': len(combined_summary) / len(context),
            'compression_level': level,
            'preservation_metadata': self.extract_preservation_metadata(context, combined_summary)
        }
    
    def summarize_chunk(self, chunk: str, level: Dict) -> str:
        """Summarize individual chunk with level-specific instructions"""
        
        prompt = f"""
        Summarize the following context chunk with {level['detail']} level detail.
        Preservation priority: {level['preserve']}
        
        Key requirements:
        - Preserve all {level['preserve']} information
        - Maintain causal relationships between events
        - Keep specific data points and metrics
        - Preserve decision rationales
        - Maintain temporal ordering
        
        Context chunk:
        {chunk}
        
        Summary:"""
        
        return self.llm.generate(prompt, max_tokens=len(chunk) // level['ratio'])
```

### 2. Key Moment Extraction

**Concept**: Identify and preserve the most important moments in context history.

```python
class KeyMomentExtractor:
    """Extract and preserve critical moments from context history"""
    
    def __init__(self):
        self.importance_indicators = {
            'decision_made': 0.9,
            'error_occurred': 0.8,
            'goal_achieved': 0.9,
            'strategy_changed': 0.85,
            'new_information': 0.7,
            'user_feedback': 0.8,
            'breakthrough_moment': 0.95,
            'tool_execution': 0.6,
            'state_change': 0.7,
            'preference_learned': 0.75
        }
    
    def extract_key_moments(self, conversation_history: List[Dict]) -> Dict:
        """Extract key moments using multi-criteria analysis"""
        
        key_moments = []
        routine_interactions = []
        
        for i, interaction in enumerate(conversation_history):
            importance_score = self.calculate_importance_score(interaction, i)
            
            if importance_score > 0.7:  # High importance threshold
                key_moment = {
                    'interaction': interaction,
                    'importance_score': importance_score,
                    'moment_type': self.classify_moment_type(interaction),
                    'context_impact': self.assess_context_impact(interaction, i),
                    'preservation_priority': self.calculate_preservation_priority(interaction),
                    'relationships': self.find_related_moments(interaction, conversation_history)
                }
                key_moments.append(key_moment)
            else:
                routine_interactions.append({
                    'interaction': interaction,
                    'importance_score': importance_score,
                    'summary': self.create_brief_summary(interaction)
                })
        
        return {
            'key_moments': key_moments,
            'routine_summary': self.summarize_routine_interactions(routine_interactions),
            'extraction_metadata': {
                'total_interactions': len(conversation_history),
                'key_moments_count': len(key_moments),
                'compression_achieved': 1 - (len(key_moments) / len(conversation_history))
            }
        }
    
    def calculate_importance_score(self, interaction: Dict, position: int) -> float:
        """Calculate importance score using multiple factors"""
        
        base_score = 0.3  # Baseline importance
        
        # Factor 1: Content analysis
        content_score = self.analyze_content_importance(interaction.get('content', ''))
        
        # Factor 2: Decision indicators
        decision_score = self.detect_decision_indicators(interaction)
        
        # Factor 3: Outcome significance
        outcome_score = self.assess_outcome_significance(interaction)
        
        # Factor 4: User engagement level
        engagement_score = self.measure_user_engagement(interaction)
        
        # Factor 5: Temporal significance (recent interactions slightly more important)
        temporal_score = self.calculate_temporal_significance(position)
        
        # Factor 6: Tool usage significance
        tool_score = self.assess_tool_usage_importance(interaction)
        
        # Weighted combination
        importance_score = (
            base_score +
            content_score * 0.25 +
            decision_score * 0.30 +
            outcome_score * 0.20 +
            engagement_score * 0.15 +
            temporal_score * 0.05 +
            tool_score * 0.05
        )
        
        return min(importance_score, 1.0)  # Cap at 1.0
    
    def analyze_content_importance(self, content: str) -> float:
        """Analyze content for importance indicators"""
        importance_keywords = {
            'decided': 0.3, 'concluded': 0.25, 'discovered': 0.3,
            'error': 0.25, 'failed': 0.2, 'succeeded': 0.2,
            'important': 0.15, 'critical': 0.2, 'urgent': 0.15,
            'learned': 0.2, 'realized': 0.2, 'found': 0.15,
            'changed': 0.15, 'updated': 0.1, 'modified': 0.1
        }
        
        content_lower = content.lower()
        score = 0.0
        
        for keyword, weight in importance_keywords.items():
            if keyword in content_lower:
                score += weight
        
        # Look for specific patterns
        if '?' in content:  # Questions often indicate important moments
            score += 0.1
        
        if any(word in content_lower for word in ['because', 'therefore', 'thus', 'so']):
            score += 0.15  # Reasoning indicators
        
        return min(score, 0.5)  # Cap content score contribution
```

### 3. Semantic Clustering

**Concept**: Group related context items for efficient storage and retrieval.

```python
class SemanticContextClusterer:
    """Cluster related context items using semantic similarity"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.75
        self.max_cluster_size = 10
    
    def cluster_context_items(self, context_items: List[Dict]) -> Dict:
        """Cluster context items by semantic similarity"""
        
        # Generate embeddings for all items
        embeddings = []
        for item in context_items:
            embedding = self.embedding_model.embed(item['content'])
            embeddings.append(embedding)
        
        # Perform semantic clustering
        clusters = self.perform_clustering(embeddings, context_items)
        
        # Create cluster summaries
        cluster_summaries = []
        for cluster in clusters:
            summary = self.create_cluster_summary(cluster)
            cluster_summaries.append(summary)
        
        return {
            'clusters': clusters,
            'cluster_summaries': cluster_summaries,
            'original_item_count': len(context_items),
            'cluster_count': len(clusters),
            'compression_ratio': len(cluster_summaries) / len(context_items)
        }
    
    def create_cluster_summary(self, cluster: List[Dict]) -> Dict:
        """Create comprehensive summary of clustered items"""
        
        # Extract key themes
        themes = self.extract_cluster_themes(cluster)
        
        # Identify most representative item
        representative_item = self.find_representative_item(cluster)
        
        # Create temporal summary
        temporal_info = self.extract_temporal_patterns(cluster)
        
        # Generate unified summary
        unified_summary = self.generate_unified_summary(cluster, themes)
        
        return {
            'summary': unified_summary,
            'themes': themes,
            'representative_item': representative_item,
            'temporal_info': temporal_info,
            'item_count': len(cluster),
            'importance_score': sum(item.get('importance', 0.5) for item in cluster) / len(cluster),
            'cluster_id': str(uuid.uuid4()),
            'created_at': datetime.now()
        }
```

### 4. Temporal Compression

**Concept**: Organize and compress context based on temporal patterns.

```python
class TemporalContextCompressor:
    """Compress context based on temporal patterns and recency"""
    
    def __init__(self):
        self.time_windows = {
            'immediate': timedelta(minutes=5),
            'recent': timedelta(hours=1), 
            'session': timedelta(hours=6),
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'historical': timedelta(days=30)
        }
        
        self.compression_strategies = {
            'immediate': 'preserve_all',
            'recent': 'selective_compression',
            'session': 'moderate_compression', 
            'daily': 'aggressive_compression',
            'weekly': 'summary_only',
            'historical': 'key_moments_only'
        }
    
    def compress_by_temporal_windows(self, context_history: List[Dict]) -> Dict:
        """Apply temporal-based compression strategies"""
        
        current_time = datetime.now()
        temporal_buckets = {window: [] for window in self.time_windows.keys()}
        
        # Bucket items by temporal windows
        for item in context_history:
            item_time = datetime.fromisoformat(item['timestamp'])
            time_diff = current_time - item_time
            
            # Assign to most specific applicable window
            for window, duration in self.time_windows.items():
                if time_diff <= duration:
                    temporal_buckets[window].append(item)
                    break
        
        # Apply compression strategy for each window
        compressed_windows = {}
        for window, items in temporal_buckets.items():
            if not items:
                continue
                
            strategy = self.compression_strategies[window]
            compressed_windows[window] = self.apply_compression_strategy(items, strategy, window)
        
        return {
            'compressed_windows': compressed_windows,
            'original_item_count': len(context_history),
            'compression_summary': self.create_compression_summary(compressed_windows),
            'temporal_distribution': {k: len(v) for k, v in temporal_buckets.items()}
        }
    
    def apply_compression_strategy(self, items: List[Dict], strategy: str, window: str) -> Dict:
        """Apply specific compression strategy to temporal window"""
        
        if strategy == 'preserve_all':
            return {
                'strategy': strategy,
                'items': items,
                'compression_ratio': 1.0,
                'summary': f"Preserved all {len(items)} items from {window} window"
            }
        
        elif strategy == 'selective_compression':
            # Keep high-importance items, compress others
            high_importance = [item for item in items if item.get('importance', 0.5) > 0.7]
            low_importance = [item for item in items if item.get('importance', 0.5) <= 0.7]
            
            compressed_low = self.create_summary(low_importance) if low_importance else None
            
            return {
                'strategy': strategy,
                'preserved_items': high_importance,
                'compressed_summary': compressed_low,
                'compression_ratio': (len(high_importance) + (1 if compressed_low else 0)) / len(items),
                'summary': f"Preserved {len(high_importance)} high-importance items, compressed {len(low_importance)}"
            }
        
        elif strategy == 'key_moments_only':
            key_moments = [item for item in items if item.get('importance', 0.5) > 0.8]
            
            return {
                'strategy': strategy,
                'key_moments': key_moments,
                'compression_ratio': len(key_moments) / len(items) if items else 0,
                'summary': f"Extracted {len(key_moments)} key moments from {len(items)} items"
            }
        
        else:
            # Default to summary-only
            summary = self.create_comprehensive_summary(items)
            return {
                'strategy': 'summary_only',
                'summary': summary,
                'compression_ratio': 1 / len(items) if items else 0,
                'summary_metadata': f"Compressed {len(items)} items into single summary"
            }
```

## Trade-offs and Optimization

### Critical Trade-offs

1. **Compression Ratio vs Information Loss**
   - Higher compression = more space savings
   - Higher compression = potential information loss
   - Optimal range: 5:1 to 10:1 for most applications

2. **Processing Speed vs Compression Quality**
   - Better compression requires more processing time
   - Real-time applications need fast compression
   - Batch processing allows more sophisticated algorithms

3. **Memory Usage vs Retrieval Speed**
   - More compressed = less memory usage
   - More compressed = potentially slower retrieval
   - Balance depends on application requirements

4. **Context Freshness vs Historical Depth**
   - Recent context is more relevant
   - Historical context provides valuable patterns
   - Aging strategies help balance both needs

### Optimization Strategies

#### 1. Adaptive Compression

```python
class AdaptiveContextCompressor:
    """Dynamically adjust compression based on context characteristics"""
    
    def __init__(self):
        self.compression_profiles = {
            'conversation': {'ratio': 5, 'preserve_decisions': True},
            'technical': {'ratio': 3, 'preserve_code': True},
            'research': {'ratio': 7, 'preserve_citations': True},
            'support': {'ratio': 4, 'preserve_issues': True}
        }
    
    def compress_adaptively(self, context: str, context_type: str = None) -> Dict:
        """Apply adaptive compression based on context analysis"""
        
        # Analyze context if type not provided
        if not context_type:
            context_type = self.analyze_context_type(context)
        
        # Get appropriate compression profile
        profile = self.compression_profiles.get(context_type, self.compression_profiles['conversation'])
        
        # Apply context-aware compression
        return self.compress_with_profile(context, profile)
```

#### 2. Quality-Aware Compression

```python
class QualityAwareCompressor:
    """Monitor compression quality and adjust algorithms accordingly"""
    
    def __init__(self):
        self.quality_metrics = ['preservation_score', 'retrieval_accuracy', 'coherence_score']
        self.quality_targets = {'preservation_score': 0.9, 'retrieval_accuracy': 0.95, 'coherence_score': 0.85}
        
    def compress_with_quality_monitoring(self, context: str) -> Dict:
        """Apply compression with continuous quality monitoring"""
        
        compressed = self.apply_compression(context)
        quality_scores = self.evaluate_compression_quality(context, compressed)
        
        # Adjust compression if quality targets not met
        if not self.meets_quality_targets(quality_scores):
            compressed = self.adjust_compression_for_quality(context, compressed, quality_scores)
        
        return {
            'compressed_context': compressed,
            'quality_scores': quality_scores,
            'meets_targets': self.meets_quality_targets(quality_scores)
        }
```

## Key Takeaways

1. **Information Density Focus**: Maximize useful information per token
2. **Multi-Algorithm Approach**: Different algorithms excel in different scenarios
3. **Quality Metrics**: Monitor compression effectiveness continuously
4. **Adaptive Strategies**: Adjust compression based on context characteristics
5. **Trade-off Awareness**: Balance compression ratio with information preservation

Context compression is both an art and a science, requiring careful balance between competing objectives while maintaining system reliability and performance.