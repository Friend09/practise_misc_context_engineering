"""
Exercise 1.1: Basic Context Structuring

Objective: Create well-structured context for different AI tasks
Skills: Context organization, JSON/XML formatting, validation
Duration: 60 minutes

This exercise teaches you to structure context effectively using the 9 core components
of context engineering. You'll practice organizing context for different scenarios
and validate your structures.
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
            
        TODO: Implement this method
        - Validate the component type
        - Check for duplicate component types (should replace if exists)
        - Add component to self.components dictionary
        - Return True if successful, False otherwise
        """
        # Your implementation here
        pass
    
    def validate_structure(self) -> Dict[str, Any]:
        """
        Validate the complete context structure.
        
        Returns:
            Dict containing validation results with:
            - is_valid: bool
            - errors: List[str]
            - warnings: List[str]
            - completeness_score: float (0.0 to 1.0)
        
        TODO: Implement validation logic
        - Check if required components are present
        - Validate component content
        - Calculate completeness score based on component quality
        - Return comprehensive validation results
        """
        # Your implementation here
        pass
    
    def to_json(self) -> str:
        """
        Convert context structure to JSON format.
        
        Returns:
            str: JSON representation of the context structure
            
        TODO: Implement JSON serialization
        - Convert all components to dictionaries
        - Include metadata
        - Handle datetime serialization
        - Return properly formatted JSON string
        """
        # Your implementation here
        pass
    
    def to_xml(self) -> str:
        """
        Convert context structure to XML format.
        
        Returns:
            str: XML representation of the context structure
            
        TODO: Implement XML serialization
        - Create XML structure with proper hierarchy
        - Convert each component to XML elements
        - Include metadata as attributes
        - Return properly formatted XML string
        """
        # Your implementation here
        pass
    
    def from_json(self, json_str: str) -> bool:
        """
        Load context structure from JSON.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            bool: True if loaded successfully
            
        TODO: Implement JSON deserialization
        - Parse JSON string
        - Reconstruct component objects
        - Handle datetime deserialization
        - Validate loaded structure
        """
        # Your implementation here
        pass


class ContextScenarioBuilder:
    """Helper class for building context scenarios"""
    
    @staticmethod
    def build_code_review_scenario() -> ContextStructure:
        """
        Build a context structure for a code review scenario.
        
        Returns:
            ContextStructure: Complete context for code review task
            
        TODO: Create a realistic code review context
        - System instructions for a code review assistant
        - User input requesting a specific code review
        - Short-term memory of recent code discussions
        - Any other relevant components
        """
        # Your implementation here
        pass
    
    @staticmethod
    def build_customer_support_scenario() -> ContextStructure:
        """
        Build a context structure for customer support scenario.
        
        Returns:
            ContextStructure: Complete context for customer support task
            
        TODO: Create a realistic customer support context
        - System instructions for a support agent
        - User input with a customer issue
        - Long-term memory of customer history
        - Any other relevant components
        """
        # Your implementation here
        pass
    
    @staticmethod
    def build_research_assistant_scenario() -> ContextStructure:
        """
        Build a context structure for research assistant scenario.
        
        Returns:
            ContextStructure: Complete context for research task
            
        TODO: Create a realistic research assistant context
        - System instructions for research capabilities
        - User input with research request
        - Knowledge base information
        - Any other relevant components
        """
        # Your implementation here
        pass


def test_context_structure():
    """Test function for context structure implementation"""
    print("Testing Context Structure Implementation")
    print("=" * 50)
    
    # Test 1: Component Addition
    structure = ContextStructure()
    
    # Create system instructions
    system = SystemInstructions(
        component_type="",  # Will be set in __post_init__
        created_at=datetime.now(),
        component_id="",
        role="Software Development Assistant",
        capabilities=["code_analysis", "bug_detection", "optimization"],
        constraints=["no_destructive_operations", "explain_reasoning"],
        behavior_guidelines=["be_helpful", "be_accurate", "be_concise"],
        output_format={"format": "markdown", "include_examples": True}
    )
    
    # Test component addition
    success = structure.add_component(system)
    print(f"Component addition test: {'PASS' if success else 'FAIL'}")
    
    # Test 2: Structure Validation
    validation = structure.validate_structure()
    print(f"Structure validation test: {'PASS' if isinstance(validation, dict) else 'FAIL'}")
    
    # Test 3: JSON Serialization
    try:
        json_output = structure.to_json()
        json_test = json.loads(json_output) if json_output else None
        print(f"JSON serialization test: {'PASS' if json_test else 'FAIL'}")
    except:
        print("JSON serialization test: FAIL")
    
    # Test 4: XML Serialization
    try:
        xml_output = structure.to_xml()
        xml_test = ET.fromstring(xml_output) if xml_output else None
        print(f"XML serialization test: {'PASS' if xml_test is not None else 'FAIL'}")
    except:
        print("XML serialization test: FAIL")
    
    # Test 5: Scenario Building
    scenarios = [
        ContextScenarioBuilder.build_code_review_scenario(),
        ContextScenarioBuilder.build_customer_support_scenario(),
        ContextScenarioBuilder.build_research_assistant_scenario()
    ]
    
    valid_scenarios = [s for s in scenarios if s is not None]
    print(f"Scenario building test: {'PASS' if len(valid_scenarios) == 3 else 'FAIL'}")
    
    print("\nExercise 1.1 Implementation Status:")
    print("- Implement all TODO methods in the classes above")
    print("- Run this test function to validate your implementation")
    print("- All tests should pass before moving to the next exercise")


def main():
    """Main function to run the exercise"""
    print("Context Engineering Exercise 1.1: Basic Context Structuring")
    print("=" * 60)
    print()
    
    print("This exercise focuses on:")
    print("1. Understanding the 9 context components")
    print("2. Creating structured context representations")
    print("3. Validating context completeness and quality")
    print("4. Serializing context to different formats")
    print("5. Building realistic context scenarios")
    print()
    
    print("Your tasks:")
    print("1. Implement all methods marked with TODO")
    print("2. Create realistic context scenarios")
    print("3. Test your implementation thoroughly")
    print("4. Ensure all validation tests pass")
    print()
    
    # Run tests
    test_context_structure()
    
    print("\nNext Steps:")
    print("- Complete all TODO implementations")
    print("- Experiment with different context structures")
    print("- Try creating your own scenario types")
    print("- Move to Exercise 1.2 when ready")


if __name__ == "__main__":
    main()