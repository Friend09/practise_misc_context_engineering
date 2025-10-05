"""
Exercise 1.1 Solution: Basic Context Structuring

This solution demonstrates proper implementation of context structuring
using the 9 core components of context engineering.
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid


@dataclass
class ContextComponent:
    """Base class for all context components"""
    component_type: str
    created_at: datetime
    component_id: str
    
    def __post_init__(self):
        if not hasattr(self, 'component_id') or not self.component_id:
            self.component_id = str(uuid.uuid4())
        if not hasattr(self, 'created_at') or not self.created_at:
            self.created_at = datetime.now()


@dataclass
class SystemInstructions(ContextComponent):
    """Component 1: System prompts and instructions"""
    role: str
    capabilities: List[str]
    constraints: List[str]
    behavior_guidelines: List[str]
    output_format: Dict[str, Any]
    
    def __post_init__(self):
        self.component_type = "system_instructions"
        super().__post_init__()


@dataclass
class UserInput(ContextComponent):
    """Component 2: User input and queries"""
    query: str
    requirements: List[str]
    context_clues: Dict[str, Any]
    implicit_assumptions: List[str]
    
    def __post_init__(self):
        self.component_type = "user_input"
        super().__post_init__()


@dataclass
class ShortTermMemory(ContextComponent):
    """Component 3: Short-term memory and chat history"""
    recent_interactions: List[Dict[str, Any]]
    conversation_flow: List[str]
    decisions_made: List[Dict[str, Any]]
    
    def __post_init__(self):
        self.component_type = "short_term_memory"
        super().__post_init__()


@dataclass
class LongTermMemory(ContextComponent):
    """Component 4: Long-term memory and persistence"""
    user_preferences: Dict[str, Any]
    historical_patterns: List[Dict[str, Any]]
    accumulated_knowledge: Dict[str, Any]
    
    def __post_init__(self):
        self.component_type = "long_term_memory"
        super().__post_init__()


class ContextStructure:
    """Main class for organizing and validating context structures"""
    
    def __init__(self):
        self.components = {}
        self.metadata = {
            'created_at': datetime.now(),
            'structure_id': str(uuid.uuid4()),
            'version': '1.0'
        }
    
    def add_component(self, component: ContextComponent) -> bool:
        """
        Add a context component to the structure.
        
        Args:
            component: A ContextComponent instance
            
        Returns:
            bool: True if component was added successfully
        """
        try:
            # Validate component
            if not isinstance(component, ContextComponent):
                return False
            
            if not hasattr(component, 'component_type') or not component.component_type:
                return False
            
            # Add component (replace if exists)
            self.components[component.component_type] = component
            return True
            
        except Exception:
            return False
    
    def validate_structure(self) -> Dict[str, Any]:
        """
        Validate the complete context structure.
        
        Returns:
            Dict containing validation results
        """
        errors = []
        warnings = []
        
        # Required components for basic validation
        required_components = ['system_instructions', 'user_input']
        optional_components = [
            'short_term_memory', 'long_term_memory', 'knowledge_base',
            'tool_definitions', 'tool_responses', 'structured_output', 'global_state'
        ]
        
        # Check required components
        missing_required = [comp for comp in required_components 
                          if comp not in self.components]
        if missing_required:
            errors.extend([f"Missing required component: {comp}" 
                          for comp in missing_required])
        
        # Check optional components
        missing_optional = [comp for comp in optional_components 
                           if comp not in self.components]
        if missing_optional:
            warnings.extend([f"Missing optional component: {comp}" 
                            for comp in missing_optional])
        
        # Validate component content
        for comp_type, component in self.components.items():
            if comp_type == 'system_instructions':
                if not component.role:
                    errors.append("System instructions missing role definition")
                if not component.capabilities:
                    warnings.append("System instructions missing capabilities")
            
            elif comp_type == 'user_input':
                if not component.query:
                    errors.append("User input missing query")
        
        # Calculate completeness score
        total_components = len(required_components) + len(optional_components)
        present_components = len(self.components)
        completeness_score = min(present_components / total_components, 1.0)
        
        # Bonus for quality (no errors)
        if not errors:
            completeness_score = min(completeness_score + 0.1, 1.0)
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'completeness_score': completeness_score,
            'component_count': present_components,
            'total_possible_components': total_components
        }
    
    def to_json(self) -> str:
        """
        Convert context structure to JSON format.
        
        Returns:
            str: JSON representation of the context structure
        """
        try:
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            # Convert components to dictionaries
            serializable_components = {}
            for comp_type, component in self.components.items():
                serializable_components[comp_type] = asdict(component)
            
            # Create full structure
            full_structure = {
                'metadata': {
                    'created_at': self.metadata['created_at'].isoformat(),
                    'structure_id': self.metadata['structure_id'],
                    'version': self.metadata['version']
                },
                'components': serializable_components
            }
            
            return json.dumps(full_structure, indent=2, default=serialize_datetime)
            
        except Exception as e:
            return f"JSON serialization error: {str(e)}"
    
    def to_xml(self) -> str:
        """
        Convert context structure to XML format.
        
        Returns:
            str: XML representation of the context structure
        """
        try:
            # Create root element
            root = ET.Element("context_structure")
            root.set("structure_id", self.metadata['structure_id'])
            root.set("created_at", self.metadata['created_at'].isoformat())
            root.set("version", self.metadata['version'])
            
            # Add components
            components_elem = ET.SubElement(root, "components")
            
            for comp_type, component in self.components.items():
                comp_elem = ET.SubElement(components_elem, "component")
                comp_elem.set("type", comp_type)
                comp_elem.set("id", component.component_id)
                comp_elem.set("created_at", component.created_at.isoformat())
                
                # Add component data
                comp_dict = asdict(component)
                for key, value in comp_dict.items():
                    if key in ['component_id', 'created_at', 'component_type']:
                        continue  # Already added as attributes
                    
                    elem = ET.SubElement(comp_elem, key)
                    if isinstance(value, (list, dict)):
                        elem.text = json.dumps(value)
                    else:
                        elem.text = str(value)
            
            # Convert to string
            ET.indent(root, space="  ")
            return ET.tostring(root, encoding='unicode')
            
        except Exception as e:
            return f"XML serialization error: {str(e)}"
    
    def from_json(self, json_str: str) -> bool:
        """
        Load context structure from JSON.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            data = json.loads(json_str)
            
            # Load metadata
            if 'metadata' in data:
                metadata = data['metadata']
                self.metadata = {
                    'created_at': datetime.fromisoformat(metadata['created_at']),
                    'structure_id': metadata['structure_id'],
                    'version': metadata['version']
                }
            
            # Load components
            if 'components' in data:
                self.components = {}
                for comp_type, comp_data in data['components'].items():
                    # Create appropriate component type
                    if comp_type == 'system_instructions':
                        component = SystemInstructions(**comp_data)
                    elif comp_type == 'user_input':
                        component = UserInput(**comp_data)
                    elif comp_type == 'short_term_memory':
                        component = ShortTermMemory(**comp_data)
                    elif comp_type == 'long_term_memory':
                        component = LongTermMemory(**comp_data)
                    else:
                        # Generic component
                        component = ContextComponent(**comp_data)
                    
                    self.components[comp_type] = component
            
            return True
            
        except Exception:
            return False


class ContextScenarioBuilder:
    """Helper class for building context scenarios"""
    
    @staticmethod
    def build_code_review_scenario() -> ContextStructure:
        """
        Build a context structure for a code review scenario.
        
        Returns:
            ContextStructure: Complete context for code review task
        """
        structure = ContextStructure()
        
        # System instructions
        system = SystemInstructions(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            role="Senior Code Review Assistant",
            capabilities=[
                "code_analysis", "bug_detection", "security_review",
                "performance_optimization", "best_practices_enforcement"
            ],
            constraints=[
                "no_code_execution", "maintain_code_style", "preserve_functionality"
            ],
            behavior_guidelines=[
                "be_constructive", "explain_reasoning", "suggest_improvements",
                "prioritize_critical_issues"
            ],
            output_format={
                "format": "structured_review",
                "sections": ["summary", "critical_issues", "suggestions", "praise"],
                "include_line_numbers": True
            }
        )
        
        # User input
        user_input = UserInput(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            query="Please review this Python function for security vulnerabilities and performance issues",
            requirements=[
                "identify_security_risks", "check_performance", "validate_error_handling",
                "assess_code_style"
            ],
            context_clues={
                "code_language": "python",
                "function_complexity": "medium",
                "business_critical": True
            },
            implicit_assumptions=[
                "user_wants_detailed_feedback", "production_code_quality_expected"
            ]
        )
        
        # Short-term memory
        memory = ShortTermMemory(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            recent_interactions=[
                {
                    "user": "Can you help me review some code?",
                    "assistant": "I'd be happy to help with code review. Please share the code.",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            conversation_flow=["greeting", "code_review_request", "context_gathering"],
            decisions_made=[
                {
                    "decision": "focus_on_security_and_performance",
                    "rationale": "user_mentioned_both_explicitly",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        )
        
        structure.add_component(system)
        structure.add_component(user_input)
        structure.add_component(memory)
        
        return structure
    
    @staticmethod
    def build_customer_support_scenario() -> ContextStructure:
        """
        Build a context structure for customer support scenario.
        
        Returns:
            ContextStructure: Complete context for customer support task
        """
        structure = ContextStructure()
        
        # System instructions
        system = SystemInstructions(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            role="Customer Support Specialist",
            capabilities=[
                "issue_diagnosis", "solution_recommendation", "escalation_management",
                "empathy_response", "documentation_access"
            ],
            constraints=[
                "no_refunds_without_approval", "follow_company_policies",
                "protect_customer_privacy"
            ],
            behavior_guidelines=[
                "be_empathetic", "be_solution_focused", "be_patient", "be_professional"
            ],
            output_format={
                "format": "customer_response",
                "tone": "professional_friendly",
                "include_next_steps": True
            }
        )
        
        # User input
        user_input = UserInput(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            query="My order hasn't arrived and it's been 2 weeks. I'm very frustrated!",
            requirements=[
                "track_order_status", "provide_explanation", "offer_solution",
                "address_customer_frustration"
            ],
            context_clues={
                "emotional_state": "frustrated",
                "urgency_level": "high",
                "issue_category": "shipping_delay"
            },
            implicit_assumptions=[
                "customer_expects_immediate_resolution", "order_is_legitimately_delayed"
            ]
        )
        
        # Long-term memory
        memory = LongTermMemory(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            user_preferences={
                "communication_style": "direct",
                "preferred_contact_method": "email",
                "customer_tier": "premium"
            },
            historical_patterns=[
                {
                    "pattern": "shipping_issues_common_to_region",
                    "frequency": "monthly",
                    "resolution_success_rate": 0.95
                }
            ],
            accumulated_knowledge={
                "customer_history": "loyal_customer_3_years",
                "previous_issues": 2,
                "satisfaction_rating": 4.5
            }
        )
        
        structure.add_component(system)
        structure.add_component(user_input)
        structure.add_component(memory)
        
        return structure
    
    @staticmethod
    def build_research_assistant_scenario() -> ContextStructure:
        """
        Build a context structure for research assistant scenario.
        
        Returns:
            ContextStructure: Complete context for research task
        """
        structure = ContextStructure()
        
        # System instructions
        system = SystemInstructions(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            role="Research Assistant Specialist",
            capabilities=[
                "literature_search", "data_analysis", "source_verification",
                "synthesis_writing", "citation_management"
            ],
            constraints=[
                "use_credible_sources_only", "cite_all_references", "avoid_plagiarism",
                "maintain_academic_standards"
            ],
            behavior_guidelines=[
                "be_thorough", "be_objective", "be_accurate", "be_systematic"
            ],
            output_format={
                "format": "academic_report",
                "citation_style": "APA",
                "include_bibliography": True,
                "structure": ["abstract", "introduction", "findings", "conclusion"]
            }
        )
        
        # User input
        user_input = UserInput(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            query="Research the impact of AI on software engineering productivity in 2023-2024",
            requirements=[
                "find_recent_studies", "analyze_productivity_metrics",
                "identify_key_trends", "compare_different_AI_tools"
            ],
            context_clues={
                "research_depth": "comprehensive",
                "time_frame": "recent_2_years",
                "domain": "software_engineering"
            },
            implicit_assumptions=[
                "academic_quality_expected", "quantitative_data_preferred"
            ]
        )
        
        # Short-term memory
        memory = ShortTermMemory(
            component_type="",
            created_at=datetime.now(),
            component_id="",
            recent_interactions=[
                {
                    "user": "I need help with research on AI productivity",
                    "assistant": "I can help with comprehensive research. What specific aspects?",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            conversation_flow=["research_request", "scope_clarification", "methodology_discussion"],
            decisions_made=[
                {
                    "decision": "focus_on_empirical_studies",
                    "rationale": "user_needs_quantitative_evidence",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        )
        
        structure.add_component(system)
        structure.add_component(user_input)
        structure.add_component(memory)
        
        return structure


def demonstrate_solution():
    """Demonstrate the complete solution"""
    print("Context Engineering Exercise 1.1 Solution Demonstration")
    print("=" * 60)
    
    # Test 1: Build and validate code review scenario
    print("\n1. Code Review Scenario:")
    print("-" * 30)
    code_review = ContextScenarioBuilder.build_code_review_scenario()
    validation = code_review.validate_structure()
    
    print(f"Valid structure: {validation['is_valid']}")
    print(f"Completeness score: {validation['completeness_score']:.2f}")
    print(f"Components: {validation['component_count']}/{validation['total_possible_components']}")
    if validation['warnings']:
        print(f"Warnings: {len(validation['warnings'])}")
    
    # Test 2: JSON serialization
    print("\n2. JSON Serialization:")
    print("-" * 30)
    json_output = code_review.to_json()
    print(f"JSON length: {len(json_output)} characters")
    print("Sample JSON structure:")
    try:
        json_data = json.loads(json_output)
        print(f"- Metadata: {list(json_data.get('metadata', {}).keys())}")
        print(f"- Components: {list(json_data.get('components', {}).keys())}")
    except:
        print("Error parsing JSON")
    
    # Test 3: XML serialization
    print("\n3. XML Serialization:")
    print("-" * 30)
    xml_output = code_review.to_xml()
    print(f"XML length: {len(xml_output)} characters")
    try:
        root = ET.fromstring(xml_output)
        print(f"Root element: {root.tag}")
        print(f"Components count: {len(root.find('components'))}")
    except:
        print("Error parsing XML")
    
    # Test 4: Load from JSON
    print("\n4. JSON Round-trip Test:")
    print("-" * 30)
    new_structure = ContextStructure()
    success = new_structure.from_json(json_output)
    print(f"Load from JSON: {'SUCCESS' if success else 'FAILED'}")
    if success:
        new_validation = new_structure.validate_structure()
        print(f"Loaded structure valid: {new_validation['is_valid']}")
    
    # Test 5: All scenarios
    print("\n5. All Scenario Types:")
    print("-" * 30)
    scenarios = {
        "Code Review": ContextScenarioBuilder.build_code_review_scenario(),
        "Customer Support": ContextScenarioBuilder.build_customer_support_scenario(),
        "Research Assistant": ContextScenarioBuilder.build_research_assistant_scenario()
    }
    
    for name, scenario in scenarios.items():
        validation = scenario.validate_structure()
        print(f"{name}: Valid={validation['is_valid']}, "
              f"Score={validation['completeness_score']:.2f}, "
              f"Components={validation['component_count']}")
    
    print("\n" + "=" * 60)
    print("Solution demonstrates:")
    print("✅ Proper context component structure")
    print("✅ Comprehensive validation logic")
    print("✅ JSON and XML serialization")
    print("✅ Realistic scenario building")
    print("✅ Error handling and robustness")


if __name__ == "__main__":
    demonstrate_solution()