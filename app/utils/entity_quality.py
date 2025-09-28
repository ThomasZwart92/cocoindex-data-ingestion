"""
Entity Quality Validation Module

Provides comprehensive validation and quality scoring for extracted entities.
Filters out low-quality entities like verb phrases, generic terms, and fragments.
"""

import re
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EntityQualityValidator:
    """Validates and scores entity quality to filter out non-entities"""
    
    # Generic words that are too vague to be useful entities
    GENERIC_STOPWORDS = {
        # Single generic words
        'issue', 'problem', 'solution', 'system', 'process', 'data',
        'information', 'status', 'error', 'result', 'update', 'change',
        'last', 'first', 'next', 'current', 'new', 'old', 'previous',
        'thing', 'stuff', 'item', 'object', 'element', 'part',
        'way', 'method', 'approach', 'technique',
        # Temporal references
        'yesterday', 'today', 'tomorrow', 'now', 'then', 'later',
        'earlier', 'before', 'after', 'recently', 'soon',
        # Pronouns and determiners
        'it', 'this', 'that', 'these', 'those', 'them', 'they',
        'he', 'she', 'we', 'you', 'i', 'me', 'us',
        'some', 'any', 'all', 'none', 'each', 'every',
        # Too vague
        'one', 'two', 'three', 'many', 'few', 'several', 'various',
        'different', 'same', 'other', 'another',
    }
    
    # Verb patterns that indicate phrases rather than entities
    VERB_INDICATORS = [
        'did', 'does', 'doing', 'done', 'do',
        'was', 'were', 'been', 'being', 'be', 'is', 'are', 'am',
        'has', 'have', 'had', 'having',
        'will', 'would', 'could', 'should', 'might', 'may', 'can',
        'causing', 'caused', 'causes', 'cause',
        'making', 'made', 'makes', 'make',
        'going', 'went', 'goes', 'go',
        'coming', 'came', 'comes', 'come',
        'getting', 'got', 'gets', 'get',
        'taking', 'took', 'takes', 'take',
        'giving', 'gave', 'gives', 'give',
        'using', 'used', 'uses', 'use',
        'finding', 'found', 'finds', 'find',
        'working', 'worked', 'works', 'work',
        'trying', 'tried', 'tries', 'try',
        'need', 'needs', 'needed', 'needing',
        'want', 'wants', 'wanted', 'wanting',
    ]
    
    # Question words that indicate fragments rather than entities
    QUESTION_INDICATORS = [
        'what', 'where', 'when', 'why', 'how', 'who', 'whom', 'which',
        'whose', 'whether', 'wherever', 'whenever',
    ]
    
    # Known good acronyms and abbreviations to preserve
    KNOWN_ACRONYMS = {
        # Technical
        'API', 'UI', 'UX', 'URL', 'URI', 'HTML', 'CSS', 'JSON', 'XML',
        'SQL', 'CPU', 'GPU', 'RAM', 'ROM', 'SSD', 'HDD', 'USB', 'HDMI',
        'LED', 'LCD', 'OLED', 'PCB', 'IC', 'AC', 'DC', 'RF', 'EMI',
        # Business/Process
        'CEO', 'CTO', 'CFO', 'VP', 'HR', 'IT', 'QA', 'QC',
        'RCA', 'CAPA', 'SOP', 'KPI', 'ROI', 'TCO', 'SLA',
        'ERP', 'CRM', 'SCM', 'BPM', 'BI',
        # Standards
        'ISO', 'IEEE', 'ANSI', 'DIN', 'JIS',
        # Common
        'USA', 'UK', 'EU', 'UN', 'WHO', 'FDA', 'EPA', 'FCC',
        'GPS', 'PDF', 'ZIP', 'FTP', 'HTTP', 'HTTPS', 'SMTP',
    }
    
    # Domain-specific terms that might look generic but are important
    DOMAIN_EXCEPTIONS = {
        'firmware', 'software', 'hardware', 'database', 'server',
        'client', 'user', 'admin', 'administrator', 'operator',
        'sensor', 'actuator', 'controller', 'processor',
        'configuration', 'setting', 'parameter', 'variable',
        'interface', 'protocol', 'standard', 'specification',
        'procedure', 'process', 'workflow', 'inspection', 'cleaning',
        'de-airing', 'tank', 'chlorine', 'visit', 'maintenance',
    }
    
    @staticmethod
    def is_valid_entity(name: str, entity_type: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if a name represents a valid entity.
        
        Args:
            name: The entity name to validate
            entity_type: Optional entity type for context
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not name or not name.strip():
            return False, "empty_name"
        
        name_lower = name.lower().strip()
        words = name_lower.split()
        
        # Check for known good acronyms
        if name.upper() in EntityQualityValidator.KNOWN_ACRONYMS:
            return True, ""
        
        # Check minimum length (with exceptions)
        if len(name_lower) < 3 and name.upper() not in EntityQualityValidator.KNOWN_ACRONYMS:
            return False, "too_short"
        
        # Check for pronouns
        if name_lower in ['it', 'this', 'that', 'these', 'those', 'them', 'they']:
            return False, "pronoun"
        
        # Check for question phrases
        if any(word in EntityQualityValidator.QUESTION_INDICATORS for word in words):
            return False, "question_phrase"
        
        # Check for verb phrases (entities should be nouns)
        first_word = words[0] if words else ""
        if first_word in EntityQualityValidator.VERB_INDICATORS:
            return False, "verb_phrase"
        
        # Check if it's just a generic stopword (unless it's a known exception)
        if (name_lower in EntityQualityValidator.GENERIC_STOPWORDS and 
            name_lower not in EntityQualityValidator.DOMAIN_EXCEPTIONS):
            # Allow if it's part of a compound term
            if len(words) == 1:
                return False, "generic_term"

        # Check for sentence fragments (too many words)
        if len(words) > 9:
            return False, "sentence_fragment"
        
        # Check for temporal references
        temporal_words = {'yesterday', 'today', 'tomorrow', 'now', 'then', 'later', 'earlier'}
        if any(word in temporal_words for word in words):
            return False, "temporal_reference"
        
        # Check for pure numbers
        if name_lower.replace(' ', '').isdigit():
            return False, "pure_number"
        
        return True, ""
    
    @staticmethod
    def calculate_quality_score(
        entity_name: str,
        entity_type: Optional[str] = None,
        confidence: float = 0.5,
        relationship_count: int = 0
    ) -> float:
        """
        Calculate a quality score for an entity.
        
        Args:
            entity_name: The entity name
            entity_type: The entity type
            confidence: Extraction confidence
            relationship_count: Number of relationships
            
        Returns:
            Quality score between 0 and 1
        """
        score = confidence  # Start with extraction confidence
        name_lower = entity_name.lower().strip()
        words = name_lower.split()
        
        # Bonus for known acronyms
        if entity_name.upper() in EntityQualityValidator.KNOWN_ACRONYMS:
            score *= 1.5
        
        # Penalty for very short names (unless acronym)
        if len(entity_name) < 3 and entity_name.upper() not in EntityQualityValidator.KNOWN_ACRONYMS:
            score *= 0.3
        elif len(entity_name) < 5:
            score *= 0.7
        
        # Penalty for generic terms
        if name_lower in EntityQualityValidator.GENERIC_STOPWORDS:
            score *= 0.3
        
        # Penalty for verb phrases
        if words and words[0] in EntityQualityValidator.VERB_INDICATORS:
            score *= 0.2
        
        # Penalty for question phrases
        if any(word in EntityQualityValidator.QUESTION_INDICATORS for word in words):
            score *= 0.1
        
        # Bonus for compound meaningful terms
        if 2 <= len(words) <= 3:
            score *= 1.2
        
        # Heavy penalty for sentence fragments
        if len(words) > 5:
            score *= 0.2
        
        # Bonus for entities with relationships
        if relationship_count > 0:
            score *= (1 + min(relationship_count * 0.1, 0.5))  # Up to 50% bonus
        
        # Bonus for specific entity types
        specific_types = {'component', 'procedure', 'specification', 'organization', 'person'}
        if entity_type and entity_type.lower() in specific_types:
            score *= 1.2
        
        # Penalty for generic types
        generic_types = {'concept', 'other', 'state'}
        if entity_type and entity_type.lower() in generic_types:
            score *= 0.8
        
        # Cap between 0 and 1
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def should_keep_entity(
        entity_name: str,
        entity_type: Optional[str] = None,
        confidence: float = 0.5,
        relationship_count: int = 0,
        min_quality_score: float = 0.4
    ) -> Tuple[bool, float, str]:
        """
        Determine if an entity should be kept based on quality criteria.
        
        Args:
            entity_name: The entity name
            entity_type: The entity type
            confidence: Extraction confidence
            relationship_count: Number of relationships
            min_quality_score: Minimum score threshold
            
        Returns:
            Tuple of (should_keep, quality_score, reason)
        """
        # First check basic validity
        is_valid, invalid_reason = EntityQualityValidator.is_valid_entity(entity_name, entity_type)
        if not is_valid:
            # Even invalid entities might be kept if they have many relationships
            if relationship_count >= 5:
                return True, 0.5, f"kept_despite_{invalid_reason}_due_to_relationships"
            return False, 0.0, invalid_reason
        
        # Calculate quality score
        quality_score = EntityQualityValidator.calculate_quality_score(
            entity_name, entity_type, confidence, relationship_count
        )
        
        # Determine if we should keep it
        if quality_score >= min_quality_score:
            return True, quality_score, "good_quality"
        elif relationship_count >= 3:
            # Keep lower quality entities if they have relationships
            return True, quality_score, "kept_for_relationships"
        else:
            return False, quality_score, "low_quality"
    
    @staticmethod
    def filter_entities(
        entities: List[Dict],
        relationships: Optional[List[Dict]] = None,
        min_quality_score: float = 0.4,
        preserve_with_relationships: int = 3
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a list of entities based on quality criteria.
        
        Args:
            entities: List of entity dictionaries
            relationships: Optional list of relationships
            min_quality_score: Minimum quality score
            preserve_with_relationships: Keep if has this many relationships
            
        Returns:
            Tuple of (kept_entities, filtered_entities)
        """
        # Count relationships per entity
        relationship_counts = {}
        if relationships:
            for rel in relationships:
                source_id = rel.get('source_entity_id')
                target_id = rel.get('target_entity_id')
                relationship_counts[source_id] = relationship_counts.get(source_id, 0) + 1
                relationship_counts[target_id] = relationship_counts.get(target_id, 0) + 1
        
        kept_entities = []
        filtered_entities = []
        
        for entity in entities:
            entity_id = entity.get('id')
            entity_name = entity.get('entity_name', entity.get('name', ''))
            entity_type = entity.get('entity_type', entity.get('type'))
            confidence = entity.get('confidence', entity.get('confidence_score', 0.5))
            rel_count = relationship_counts.get(entity_id, 0)
            
            should_keep, quality_score, reason = EntityQualityValidator.should_keep_entity(
                entity_name,
                entity_type,
                confidence,
                rel_count,
                min_quality_score
            )
            
            # Add quality metadata
            entity['quality_score'] = quality_score
            entity['quality_reason'] = reason
            
            if should_keep:
                kept_entities.append(entity)
            else:
                filtered_entities.append(entity)
                logger.info(f"Filtering entity '{entity_name}' ({entity_type}): {reason}, score={quality_score:.2f}")
        
        logger.info(f"Entity filtering: kept {len(kept_entities)}, filtered {len(filtered_entities)}")
        return kept_entities, filtered_entities
