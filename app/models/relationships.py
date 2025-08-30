"""
Knowledge Graph Relationship Definitions
Optimized for smart water dispenser company documentation
"""
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel


class RelationshipType(Enum):
    """
    14 relationship types for comprehensive knowledge graph
    Each tuple contains: (label, source_types, target_types, suggested_properties)
    """
    
    # Technical/Product Relationships (5)
    COMPONENT_OF = (
        "COMPONENT_OF",
        ["Component", "Feature", "Module"],
        ["Product", "System", "Component"],
        {
            "component_type": "Type of component (hardware/software/accessory)",
            "quantity": "Number of components",
            "optional": "Whether component is optional",
            "position": "Physical or logical position"
        }
    )
    
    CONNECTS_TO = (
        "CONNECTS_TO",
        ["Product", "Component", "System", "Software"],
        ["Product", "Component", "System", "Platform"],
        {
            "connection_type": "Type (physical/bluetooth/network/api)",
            "protocol": "Communication protocol (BLE/WiFi/HTTP/MQTT)",
            "bidirectional": "Two-way connection or one-way",
            "required": "Mandatory or optional connection",
            "max_distance": "Maximum connection distance",
            "bandwidth": "Data transfer capacity"
        }
    )
    
    DEPENDS_ON = (
        "DEPENDS_ON",
        ["Product", "Component", "Feature", "Process", "Software"],
        ["Component", "Service", "Resource", "Infrastructure"],
        {
            "dependency_type": "Type (runtime/compile/configuration)",
            "version": "Version requirement",
            "critical": "Whether dependency is critical",
            "fallback": "Alternative if unavailable",
            "performance_impact": "Impact on performance"
        }
    )
    
    REPLACES = (
        "REPLACES",
        ["Product", "Component", "Software", "Document"],
        ["Product", "Component", "Software", "Document"],
        {
            "migration_required": "Whether migration is needed",
            "backwards_compatible": "Maintains compatibility",
            "deprecation_date": "When old version deprecated",
            "reason": "Reason for replacement",
            "upgrade_path": "How to upgrade"
        }
    )
    
    TROUBLESHOOTS = (
        "TROUBLESHOOTS",
        ["Document", "Procedure", "Guide"],
        ["Issue", "Error", "Problem", "Component"],
        {
            "error_code": "Specific error code",
            "severity": "Issue severity (critical/high/medium/low)",
            "frequency": "How often issue occurs",
            "resolution_time": "Typical time to resolve",
            "success_rate": "Success rate of solution",
            "symptoms": "Observable symptoms"
        }
    )
    
    # Documentation/Knowledge Relationships (4)
    DEFINES = (
        "DEFINES",
        ["Document", "Specification", "Standard"],
        ["Concept", "Process", "Standard", "Value", "Term"],
        {
            "definition_type": "Type (technical/business/operational)",
            "authority_level": "How authoritative (canonical/suggested)",
            "scope": "Scope of definition",
            "version": "Version of definition"
        }
    )
    
    DOCUMENTS = (
        "DOCUMENTS",
        ["Document", "Guide", "Manual"],
        ["Product", "Process", "Feature", "System"],
        {
            "documentation_type": "Type (technical/user/api)",
            "detail_level": "Level of detail (overview/detailed/reference)",
            "version": "Version documented",
            "last_updated": "Last update date"
        }
    )
    
    REFERENCES = (
        "REFERENCES",
        ["Document", "Specification", "Report"],
        ["Document", "Standard", "Source", "Study"],
        {
            "reference_type": "Type (citation/link/mention)",
            "page": "Page number or section",
            "url": "External URL if applicable",
            "relevance": "How relevant to source"
        }
    )
    
    TARGETS = (
        "TARGETS",
        ["Document", "Campaign", "Feature", "Product"],
        ["Segment", "Department", "UserType", "Market"],
        {
            "segment_size": "Size of target segment",
            "priority": "Priority level (high/medium/low)",
            "strategy": "Approach strategy",
            "maturity": "Segment maturity (new/growing/mature)",
            "effectiveness": "How effective for target"
        }
    )
    
    # Business/Organizational Relationships (3)
    RESPONSIBLE_FOR = (
        "RESPONSIBLE_FOR",
        ["Department", "Team", "Person", "Role"],
        ["Product", "Process", "Component", "Customer", "Document"],
        {
            "responsibility_type": "Type (owner/maintainer/support)",
            "since": "Start date of responsibility",
            "sla": "Service level agreement",
            "contact": "Contact information",
            "escalation": "Escalation path"
        }
    )
    
    SERVES = (
        "SERVES",
        ["Product", "Feature", "Service", "Department"],
        ["Customer", "Market", "Segment", "Region"],
        {
            "service_type": "Type of service",
            "tier": "Service tier (premium/standard/basic)",
            "region": "Geographic region",
            "contract_type": "Contract arrangement",
            "revenue": "Revenue from relationship"
        }
    )
    
    IMPACTS = (
        "IMPACTS",
        ["Issue", "Change", "Feature", "Decision"],
        ["Product", "Customer", "Process", "Component"],
        {
            "impact_type": "Type (positive/negative/neutral)",
            "severity": "Severity of impact",
            "timeframe": "When impact occurs",
            "measurable": "How to measure impact",
            "mitigation": "How to mitigate negative impact"
        }
    )
    
    # Flexible Relationships (2)
    RELATES_TO = (
        "RELATES_TO",
        ["Any"],  # Can connect any entity types
        ["Any"],
        {
            "relationship_type": "Specific type of relationship (REQUIRED)",
            "description": "Description of relationship",
            "strength": "Strength of relationship"
        }
    )
    
    COMPATIBLE_WITH = (
        "COMPATIBLE_WITH",
        ["Product", "Component", "Software", "Accessory"],
        ["Product", "Component", "Software", "Standard"],
        {
            "compatibility_level": "Level (full/partial/conditional)",
            "version_range": "Compatible versions",
            "requirements": "Requirements for compatibility",
            "tested": "Whether compatibility is tested",
            "certification": "Certification status"
        }
    )
    
    def __init__(self, label, source_types, target_types, suggested_props):
        self.label = label
        self.source_types = source_types
        self.target_types = target_types
        self.suggested_properties = suggested_props
    
    @classmethod
    def get_by_label(cls, label: str) -> Optional['RelationshipType']:
        """Get relationship type by label string"""
        for rel_type in cls:
            if rel_type.label == label:
                return rel_type
        return None
    
    def validate_entities(self, source_type: str, target_type: str) -> bool:
        """Validate if entity types are valid for this relationship"""
        if "Any" in self.source_types or source_type in self.source_types:
            if "Any" in self.target_types or target_type in self.target_types:
                return True
        return False


class RelationshipProperties(BaseModel):
    """Core properties that every relationship should have"""
    confidence: float  # 0.0 to 1.0
    source_text: str  # Context snippet from document
    extracted_at: datetime  # When relationship was extracted
    extraction_method: str  # "llm_gpt4", "llm_gemini", "rule_based", "manual"
    
    # Strongly recommended
    page_number: Optional[int] = None  # Page in document
    section: Optional[str] = None  # Document section
    validated: bool = False  # Human reviewed?
    validator: Optional[str] = None  # Who validated
    
    # Additional dynamic properties
    additional_properties: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class Relationship(BaseModel):
    """Complete relationship with entities and properties"""
    source_entity: str  # Entity ID or name
    source_type: str  # Entity type
    relationship_type: RelationshipType
    target_entity: str  # Entity ID or name
    target_type: str  # Entity type
    properties: RelationshipProperties
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        """Convert to properties for Cypher query"""
        props = {
            "confidence": self.properties.confidence,
            "source_text": self.properties.source_text,
            "extracted_at": self.properties.extracted_at.isoformat(),
            "extraction_method": self.properties.extraction_method,
            "validated": self.properties.validated
        }
        
        if self.properties.page_number:
            props["page_number"] = self.properties.page_number
        if self.properties.section:
            props["section"] = self.properties.section
        if self.properties.validator:
            props["validator"] = self.properties.validator
            
        # Add all additional properties
        props.update(self.properties.additional_properties)
        
        return props


# Domain-specific entity types for smart water dispenser company
ENTITY_TYPES = {
    # Products & Technical
    "Product": ["Water Dispenser", "Model", "Product Line"],
    "Component": ["Hardware", "Sensor", "Pump", "Filter", "Module"],
    "Feature": ["Flavoring", "Temperature Control", "IoT", "Display"],
    "Software": ["Firmware", "Mobile App", "Cloud Platform", "API"],
    "Issue": ["Error Code", "Problem", "Defect", "Bug"],
    "System": ["Infrastructure", "Platform", "Network"],
    
    # Organizational
    "Department": ["Sales", "Marketing", "Engineering", "Support", "Finance", "Supply Chain", "Logistics"],
    "Team": ["Subteam", "Working Group", "Committee"],
    "Person": ["Employee", "Contact", "Expert", "Manager"],
    "Role": ["Position", "Responsibility", "Function"],
    "Process": ["Workflow", "Procedure", "Method", "Protocol"],
    
    # Business
    "Customer": ["Company", "Account", "Client", "Partner"],
    "Segment": ["Market Segment", "User Type", "Vertical", "Industry"],
    "Market": ["Geographic Market", "Region", "Territory"],
    "Supplier": ["Vendor", "Partner", "Manufacturer"],
    
    # Documentation
    "Document": ["Manual", "Guide", "Specification", "Report", "Policy"],
    "Concept": ["Principle", "Value", "Standard", "Best Practice"],
    "Standard": ["Protocol", "Specification", "Requirement", "Compliance"],
    
    # Generic fallback
    "Entity": ["Generic", "Other", "Unknown"]
}


def get_entity_type(entity_name: str, entity_metadata: Dict[str, Any] = None) -> str:
    """
    Determine entity type from name and metadata
    Returns the most specific type possible
    """
    if entity_metadata and "type" in entity_metadata:
        return entity_metadata["type"]
    
    # Simple heuristic based on keywords
    name_lower = entity_name.lower()
    
    # Products
    if any(x in name_lower for x in ["model", "x500", "x600", "dispenser", "product"]):
        return "Product"
    
    # Components
    if any(x in name_lower for x in ["sensor", "pump", "filter", "module", "board", "valve"]):
        return "Component"
    
    # Software
    if any(x in name_lower for x in ["firmware", "app", "software", "api", "platform"]):
        return "Software"
    
    # Departments
    for dept in ["sales", "marketing", "engineering", "support", "finance", "supply", "logistics"]:
        if dept in name_lower:
            return "Department"
    
    # Documents
    if any(x in name_lower for x in ["document", "manual", "guide", "spec", "report"]):
        return "Document"
    
    # Default
    return "Entity"


class RelationshipModel(BaseModel):
    """Model for relationship instances in the knowledge graph"""
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    relationship_type: str
    confidence: float = 1.0
    properties: Dict[str, Any] = {}
    extracted_at: datetime = datetime.now()
    source_text: Optional[str] = None