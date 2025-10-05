# Lesson 3: Context Structuring Techniques

## Overview

Effective context structuring is fundamental to building reliable AI systems. This lesson teaches you practical techniques for organizing context information in formats that optimize both human readability and machine processing efficiency.

## Core Structuring Principles

### 1. Information Hierarchy
Structure context with clear hierarchical relationships:
- **Primary context**: Immediate task-relevant information
- **Secondary context**: Supporting background information
- **Tertiary context**: Historical or reference information

### 2. Semantic Grouping
Organize related information together:
- **Functional grouping**: Group by purpose (user info, system state, tools)
- **Temporal grouping**: Group by time relationships (recent, historical)
- **Domain grouping**: Group by subject matter (technical, business, personal)

### 3. Access Patterns
Structure based on how context will be accessed:
- **Frequently accessed**: Top-level, easy retrieval
- **Conditionally accessed**: Nested but accessible
- **Rarely accessed**: Compressed or archived

## JSON-Based Context Organization

### Basic JSON Structure

```json
{
  "context_metadata": {
    "context_id": "ctx_12345",
    "version": "1.0",
    "created_at": "2025-01-15T10:30:00Z",
    "last_updated": "2025-01-15T11:45:00Z"
  },
  "system_instructions": {
    "role": "Senior Software Engineer Assistant",
    "capabilities": ["code_analysis", "debugging", "optimization"],
    "constraints": ["no_destructive_operations", "explain_reasoning"],
    "output_format": {
      "style": "markdown",
      "include_examples": true,
      "max_length": 2000
    }
  },
  "current_context": {
    "user_query": "Help me optimize this Python function",
    "task_type": "code_optimization",
    "priority": "medium",
    "requirements": ["maintain_functionality", "improve_performance"]
  },
  "conversation_history": {
    "recent_turns": [
      {
        "turn_id": 1,
        "user": "I have a slow Python function",
        "assistant": "I'd be happy to help optimize it. Please share the code.",
        "timestamp": "2025-01-15T11:40:00Z"
      }
    ],
    "conversation_summary": "User seeking help with Python function optimization",
    "key_decisions": [
      {
        "decision": "focus_on_performance_optimization",
        "rationale": "User explicitly mentioned slow function",
        "timestamp": "2025-01-15T11:41:00Z"
      }
    ]
  },
  "knowledge_base": {
    "relevant_docs": [
      {
        "doc_id": "python_optimization_guide",
        "sections": ["loops", "data_structures", "algorithms"],
        "relevance_score": 0.95
      }
    ],
    "code_examples": [
      {
        "example_id": "list_comprehension_optimization",
        "relevance_score": 0.87
      }
    ]
  },
  "tools_available": [
    {
      "name": "code_analyzer",
      "description": "Analyze code for performance bottlenecks",
      "parameters": ["code_snippet", "analysis_type"]
    }
  ]
}
```

### Advanced JSON Patterns

#### 1. Nested Context Layers
```json
{
  "immediate_context": {
    "current_task": {...},
    "active_tools": {...}
  },
  "session_context": {
    "conversation_flow": {...},
    "user_preferences": {...}
  },
  "persistent_context": {
    "user_profile": {...},
    "learned_patterns": {...}
  }
}
```

#### 2. Context References
```json
{
  "context_references": {
    "parent_context": "ctx_12344",
    "child_contexts": ["ctx_12346", "ctx_12347"],
    "related_contexts": {
      "similar_tasks": ["ctx_11200", "ctx_11201"],
      "same_user": ["ctx_10500", "ctx_10501"]
    }
  }
}
```

## XML Context Hierarchies

### Structured XML Format

```xml
<?xml version="1.0" encoding="UTF-8"?>
<context xmlns="https://context-engineering.org/schema/v1">
  <metadata>
    <id>ctx_12345</id>
    <version>1.0</version>
    <created>2025-01-15T10:30:00Z</created>
    <updated>2025-01-15T11:45:00Z</updated>
  </metadata>
  
  <system-instructions role="Senior Software Engineer">
    <capabilities>
      <capability>code_analysis</capability>
      <capability>debugging</capability>
      <capability>optimization</capability>
    </capabilities>
    <constraints>
      <constraint>no_destructive_operations</constraint>
      <constraint>explain_reasoning</constraint>
    </constraints>
    <output-format style="markdown" include-examples="true" max-length="2000"/>
  </system-instructions>
  
  <current-context priority="medium">
    <user-query>Help me optimize this Python function</user-query>
    <task-type>code_optimization</task-type>
    <requirements>
      <requirement>maintain_functionality</requirement>
      <requirement>improve_performance</requirement>
    </requirements>
  </current-context>
  
  <conversation-history>
    <summary>User seeking help with Python function optimization</summary>
    <recent-turns>
      <turn id="1" timestamp="2025-01-15T11:40:00Z">
        <user>I have a slow Python function</user>
        <assistant>I'd be happy to help optimize it. Please share the code.</assistant>
      </turn>
    </recent-turns>
    <key-decisions>
      <decision timestamp="2025-01-15T11:41:00Z">
        <choice>focus_on_performance_optimization</choice>
        <rationale>User explicitly mentioned slow function</rationale>
      </decision>
    </key-decisions>
  </conversation-history>
</context>
```

### XML Advantages
- **Validation**: XML Schema validation ensures structure integrity
- **Namespaces**: Avoid naming conflicts in complex contexts
- **Extensibility**: Easy to add new elements without breaking structure
- **Tool support**: Rich ecosystem of XML processing tools

## Markdown Context Formatting

### Structured Markdown Format

```markdown
# Context: Python Function Optimization Session

## Metadata
- **Context ID**: ctx_12345
- **Version**: 1.0
- **Created**: 2025-01-15T10:30:00Z
- **Last Updated**: 2025-01-15T11:45:00Z

## System Instructions
**Role**: Senior Software Engineer Assistant

**Capabilities**:
- Code analysis and review
- Performance debugging
- Optimization recommendations

**Constraints**:
- No destructive operations
- Always explain reasoning
- Provide working examples

**Output Format**:
- Style: Markdown with code blocks
- Include practical examples
- Maximum length: 2000 characters

## Current Context
**User Query**: Help me optimize this Python function
**Task Type**: Code optimization
**Priority**: Medium

**Requirements**:
- [x] Maintain existing functionality
- [x] Improve performance metrics
- [ ] Preserve code readability

## Conversation History

### Recent Interaction Summary
User is seeking help optimizing a slow Python function. Initial contact established, waiting for code to be shared.

### Key Decisions Made
1. **Focus Area**: Performance optimization (11:41 AM)
   - *Rationale*: User explicitly mentioned function is slow
   - *Impact*: Will prioritize speed over other factors

### Conversation Flow
1. **Turn 1** (11:40 AM)
   - **User**: "I have a slow Python function"
   - **Assistant**: "I'd be happy to help optimize it. Please share the code."

## Knowledge Base References

### Relevant Documentation
- **Python Optimization Guide** (Relevance: 95%)
  - Sections: Loops, Data Structures, Algorithms
- **Performance Profiling Best Practices** (Relevance: 87%)

### Code Examples Available
- List comprehension optimization patterns
- Dictionary lookup vs. list iteration
- Generator expressions for memory efficiency

## Available Tools
- **Code Analyzer**: Identifies performance bottlenecks
- **Profiler**: Measures execution time and resource usage
- **Benchmark Runner**: Compares optimization results
```

### Markdown Advantages
- **Human readable**: Easy for humans to read and edit
- **Version control friendly**: Git diffs work well
- **Flexible structure**: Can adapt to different content types
- **Tool integration**: Many tools parse Markdown natively

## Hybrid Approaches and Best Practices

### Multi-Format Strategy

Use different formats for different purposes:

```python
class HybridContextManager:
    """Context manager supporting multiple formats"""
    
    def __init__(self):
        self.json_store = {}      # Machine processing
        self.xml_store = {}       # Structured validation
        self.markdown_cache = {}  # Human readability
    
    def store_context(self, context_data, formats=['json']):
        """Store context in multiple formats"""
        context_id = context_data['context_id']
        
        if 'json' in formats:
            self.json_store[context_id] = self.to_json(context_data)
        
        if 'xml' in formats:
            self.xml_store[context_id] = self.to_xml(context_data)
        
        if 'markdown' in formats:
            self.markdown_cache[context_id] = self.to_markdown(context_data)
    
    def get_context(self, context_id, preferred_format='json'):
        """Retrieve context in preferred format"""
        if preferred_format == 'json' and context_id in self.json_store:
            return self.json_store[context_id]
        elif preferred_format == 'xml' and context_id in self.xml_store:
            return self.xml_store[context_id]
        elif preferred_format == 'markdown' and context_id in self.markdown_cache:
            return self.markdown_cache[context_id]
        else:
            # Convert from available format
            return self.convert_format(context_id, preferred_format)
```

### Context Structure Optimization

#### 1. Depth vs. Breadth
- **Shallow structures**: Faster access, but may lack organization
- **Deep structures**: Better organization, but slower access
- **Balanced approach**: 3-4 levels maximum depth

#### 2. Field Naming Conventions
```python
# Good: Consistent, descriptive naming
{
    "user_query": "...",
    "user_preferences": {...},
    "user_history": [...]
}

# Bad: Inconsistent naming
{
    "query": "...",
    "userPrefs": {...},
    "UserHistory": [...]
}
```

#### 3. Data Type Consistency
```python
# Good: Consistent data types
{
    "timestamps": {
        "created_at": "2025-01-15T10:30:00Z",
        "updated_at": "2025-01-15T11:45:00Z"
    },
    "scores": {
        "relevance": 0.95,
        "confidence": 0.87
    }
}
```

## Context Validation and Testing

### JSON Schema Validation

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["context_metadata", "system_instructions", "current_context"],
  "properties": {
    "context_metadata": {
      "type": "object",
      "required": ["context_id", "version", "created_at"],
      "properties": {
        "context_id": {"type": "string", "pattern": "^ctx_[0-9]+$"},
        "version": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"}
      }
    },
    "system_instructions": {
      "type": "object",
      "required": ["role", "capabilities"],
      "properties": {
        "role": {"type": "string", "minLength": 1},
        "capabilities": {
          "type": "array",
          "items": {"type": "string"},
          "minItems": 1
        }
      }
    }
  }
}
```

### Context Quality Metrics

```python
class ContextQualityAssessor:
    """Assess context structure quality"""
    
    def assess_structure(self, context):
        """Assess overall context structure quality"""
        scores = {
            'completeness': self.check_completeness(context),
            'consistency': self.check_consistency(context),
            'efficiency': self.check_efficiency(context),
            'readability': self.check_readability(context)
        }
        
        overall_score = sum(scores.values()) / len(scores)
        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'recommendations': self.generate_recommendations(scores)
        }
    
    def check_completeness(self, context):
        """Check if all required components are present"""
        required_components = [
            'context_metadata', 'system_instructions', 'current_context'
        ]
        present_components = [comp for comp in required_components 
                            if comp in context]
        return len(present_components) / len(required_components)
    
    def check_consistency(self, context):
        """Check internal consistency of context structure"""
        # Check timestamp ordering
        # Verify reference integrity
        # Validate data type consistency
        return 0.95  # Simplified for example
```

## Best Practices Summary

### 1. Choose Format Based on Use Case
- **JSON**: Machine processing, APIs, complex nested data
- **XML**: When validation and namespaces are crucial
- **Markdown**: Human-readable documentation, version control

### 2. Design for Evolution
- Use versioning in context metadata
- Design extensible schemas
- Plan for backward compatibility

### 3. Optimize for Access Patterns
- Place frequently accessed data at top levels
- Use references for large, shared data
- Consider caching strategies

### 4. Maintain Quality
- Implement validation at context creation
- Monitor context size and complexity
- Regular cleanup of outdated context

### 5. Test Your Structures
- Validate against schemas
- Test serialization/deserialization
- Performance test with realistic data sizes

## Key Takeaways

1. **Structure Matters**: Well-organized context improves both system performance and maintainability
2. **Format Selection**: Choose formats based on primary use cases and access patterns
3. **Validation is Critical**: Always validate context structure to prevent errors
4. **Evolution Planning**: Design for change and growth in context complexity
5. **Quality Monitoring**: Continuously assess and improve context structure quality

Effective context structuring is a foundational skill that enables all other context engineering techniques to work optimally.