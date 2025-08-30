"""
Metadata Taxonomy Definitions for Document Classification
"""
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass

class DocumentCategory(str, Enum):
    """Pre-defined document categories for consistent classification"""
    
    # Technical Documentation
    PRODUCT_MANUAL = "product_manual"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    TECHNICAL_SPECIFICATION = "technical_specification"
    INSTALLATION_GUIDE = "installation_guide"
    SERVICE_MANUAL = "service_manual"
    
    # Business Documentation
    SOP = "sop"
    POLICY = "policy"
    TRAINING_MATERIAL = "training_material"
    MEETING_NOTES = "meeting_notes"
    REPORT = "report"
    
    # Customer-Facing
    FAQ = "faq"
    USER_GUIDE = "user_guide"
    RELEASE_NOTES = "release_notes"
    WARRANTY_TERMS = "warranty_terms"
    DATASHEET = "datasheet"
    
    # Internal Operations
    INCIDENT_REPORT = "incident_report"
    PROJECT_PLAN = "project_plan"
    REQUIREMENTS = "requirements"
    DESIGN_DOCUMENT = "design_document"
    TEST_PLAN = "test_plan"
    
    @classmethod
    def get_display_name(cls, category: str) -> str:
        """Get human-readable display name for category"""
        display_names = {
            "product_manual": "Product Manual",
            "troubleshooting_guide": "Troubleshooting Guide",
            "technical_specification": "Technical Specification",
            "installation_guide": "Installation Guide",
            "service_manual": "Service Manual",
            "sop": "Standard Operating Procedure",
            "policy": "Policy Document",
            "training_material": "Training Material",
            "meeting_notes": "Meeting Notes",
            "report": "Report",
            "faq": "FAQ",
            "user_guide": "User Guide",
            "release_notes": "Release Notes",
            "warranty_terms": "Warranty Terms",
            "datasheet": "Datasheet",
            "incident_report": "Incident Report",
            "project_plan": "Project Plan",
            "requirements": "Requirements Document",
            "design_document": "Design Document",
            "test_plan": "Test Plan"
        }
        return display_names.get(category, category.replace("_", " ").title())


@dataclass
class TagTaxonomy:
    """Hybrid tag taxonomy with pre-defined and extractable tags"""
    
    # Product/Model tags (auto-extracted from content)
    PRODUCT_MODELS = [
        "NC2050", "NC2068", "NC3000", "NC3100", "NC4000",
        "PC1000", "PC2000", "PC3000",
        "SM100", "SM200", "SM300"
    ]
    
    # Component tags (standardized)
    COMPONENTS = [
        # Hardware components
        "display", "screen", "lcd", "led", "oled",
        "power-supply", "psu", "battery", "adapter",
        "motherboard", "mainboard", "cpu", "processor",
        "memory", "ram", "storage", "ssd", "hdd",
        "cooling-system", "fan", "heatsink", "thermal",
        "ports", "usb", "hdmi", "ethernet", "audio",
        "keyboard", "touchpad", "mouse", "input-device",
        
        # Software components
        "firmware", "bios", "driver", "software",
        "operating-system", "os", "application"
    ]
    
    # Issue/Problem tags
    ISSUES = [
        # Display issues
        "screen-flickering", "dead-pixels", "backlight-failure",
        "color-distortion", "brightness-issue", "no-display",
        
        # Power issues
        "no-power", "power-cycling", "battery-drain",
        "charging-issue", "overheating",
        
        # Performance issues
        "slow-performance", "freezing", "crashing",
        "boot-failure", "blue-screen", "kernel-panic",
        
        # Connectivity issues
        "network-issue", "wifi-problem", "bluetooth-issue",
        "connection-drop", "port-failure"
    ]
    
    # Action/Process tags
    ACTIONS = [
        "troubleshooting", "maintenance", "calibration",
        "replacement", "upgrade", "installation",
        "configuration", "optimization", "diagnostic",
        "repair", "cleaning", "testing"
    ]
    
    # Compliance/Standards tags
    COMPLIANCE = [
        "ISO-9001", "ISO-27001", "CE", "FCC", "RoHS",
        "UL", "ETL", "Energy-Star", "GDPR", "HIPAA"
    ]
    
    # Priority/Urgency tags
    PRIORITY = [
        "critical", "high-priority", "urgent",
        "normal", "low-priority", "scheduled"
    ]
    
    @classmethod
    def get_all_predefined_tags(cls) -> List[str]:
        """Get all pre-defined tags for autocomplete"""
        all_tags = []
        all_tags.extend(cls.PRODUCT_MODELS)
        all_tags.extend(cls.COMPONENTS)
        all_tags.extend(cls.ISSUES)
        all_tags.extend(cls.ACTIONS)
        all_tags.extend(cls.COMPLIANCE)
        all_tags.extend(cls.PRIORITY)
        return sorted(list(set(all_tags)))
    
    @classmethod
    def categorize_tag(cls, tag: str) -> str:
        """Determine which category a tag belongs to"""
        tag_lower = tag.lower()
        
        if tag in cls.PRODUCT_MODELS:
            return "product"
        elif tag_lower in [t.lower() for t in cls.COMPONENTS]:
            return "component"
        elif tag_lower in [t.lower() for t in cls.ISSUES]:
            return "issue"
        elif tag_lower in [t.lower() for t in cls.ACTIONS]:
            return "action"
        elif tag in cls.COMPLIANCE:
            return "compliance"
        elif tag_lower in [t.lower() for t in cls.PRIORITY]:
            return "priority"
        else:
            return "custom"


@dataclass
class ExtractedMetadata:
    """Structure for AI-extracted metadata"""
    category: str
    tags: List[str]
    author: str = None
    department: str = None
    version: str = None
    description: str = None
    confidence_scores: Dict[str, float] = None
    extraction_model: str = "gpt-4o-mini"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "category": self.category,
            "tags": self.tags,
            "author": self.author,
            "department": self.department,
            "version": self.version,
            "description": self.description,
            "confidence_scores": self.confidence_scores,
            "extraction_model": self.extraction_model
        }