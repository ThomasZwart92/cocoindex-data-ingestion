"""
Entity Deduplication Utilities

Provides fuzzy matching and deduplication capabilities for entities.
"""
import re
from typing import List, Tuple, Dict, Set
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class EntityDeduplicator:
    """Handles entity deduplication with fuzzy matching"""
    
    # Common abbreviations and their expansions
    ABBREVIATIONS = {
        'ipa': 'isopropyl alcohol',
        'nc': 'nc',  # Keep as is for model numbers
        'lcd': 'liquid crystal display',
        'led': 'light emitting diode',
        'pcb': 'printed circuit board',
        'cpu': 'central processing unit',
        'gpu': 'graphics processing unit',
        'ram': 'random access memory',
        'rom': 'read only memory',
        'ac': 'alternating current',
        'dc': 'direct current',
        'psi': 'pounds per square inch',
        'rpm': 'revolutions per minute',
        'temp': 'temperature',
        'config': 'configuration',
        'spec': 'specification',
        'mfg': 'manufacturing',
        'mfr': 'manufacturer',
        'qty': 'quantity',
        'req': 'required',
        'min': 'minimum',
        'max': 'maximum',
        'avg': 'average',
        'std': 'standard',
        'ref': 'reference',
        'ver': 'version',
        'rev': 'revision',
        'dept': 'department',
        'mgmt': 'management',
        'admin': 'administration',
        'eng': 'engineering',
        'maint': 'maintenance',
        'ops': 'operations',
        'qa': 'quality assurance',
        'qc': 'quality control',
    }
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize entity name for comparison.
        
        Args:
            name: Entity name to normalize
            
        Returns:
            Normalized name for comparison
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common punctuation variations
        # Keep alphanumeric, spaces, and hyphens
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common suffixes/prefixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'\s+(the|a|an)$', '', normalized)
        
        return normalized
    
    @staticmethod
    def expand_abbreviations(text: str) -> str:
        """
        Expand common abbreviations in text.
        
        Args:
            text: Text containing potential abbreviations
            
        Returns:
            Text with abbreviations expanded
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        expanded_words = []
        for word in words:
            if word in EntityDeduplicator.ABBREVIATIONS:
                expanded_words.append(EntityDeduplicator.ABBREVIATIONS[word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    @staticmethod
    def calculate_similarity(name1: str, name2: str) -> float:
        """
        Calculate similarity score between two entity names.
        
        Args:
            name1: First entity name
            name2: Second entity name
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both names
        norm1 = EntityDeduplicator.normalize_name(name1)
        norm2 = EntityDeduplicator.normalize_name(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        # Check for singular/plural variations
        if EntityDeduplicator._are_singular_plural(norm1, norm2):
            return 0.98  # Very high score for singular/plural matches
        
        # Check with abbreviation expansion
        expanded1 = EntityDeduplicator.expand_abbreviations(norm1)
        expanded2 = EntityDeduplicator.expand_abbreviations(norm2)
        
        if expanded1 == expanded2:
            return 0.95  # High score for abbreviation matches
        
        # Use SequenceMatcher for fuzzy matching
        # Compare both original normalized and expanded versions
        scores = [
            SequenceMatcher(None, norm1, norm2).ratio(),
            SequenceMatcher(None, expanded1, expanded2).ratio(),
            SequenceMatcher(None, norm1, expanded2).ratio(),
            SequenceMatcher(None, expanded1, norm2).ratio(),
        ]
        
        # Take the highest similarity score
        base_score = max(scores)
        
        # Boost score if one is a substring of the other
        if norm1 in norm2 or norm2 in norm1:
            base_score = max(base_score, 0.85)
        
        # Check for common patterns (e.g., NC2056 vs NC-2056)
        if EntityDeduplicator._are_variants(norm1, norm2):
            base_score = max(base_score, 0.9)
        
        return base_score
    
    @staticmethod
    def _are_singular_plural(name1: str, name2: str) -> bool:
        """
        Check if two names are singular/plural variants of each other.
        
        Args:
            name1: First name (normalized)
            name2: Second name (normalized)
            
        Returns:
            True if they are singular/plural variants
        """
        # Simple plural rules (can be enhanced)
        if name1 + 's' == name2 or name2 + 's' == name1:
            return True
        if name1 + 'es' == name2 or name2 + 'es' == name1:
            return True
        if name1.endswith('y') and name1[:-1] + 'ies' == name2:
            return True
        if name2.endswith('y') and name2[:-1] + 'ies' == name1:
            return True
        if name1.endswith('ies') and name1[:-3] + 'y' == name2:
            return True
        if name2.endswith('ies') and name2[:-3] + 'y' == name1:
            return True
        
        # Handle compound terms with plural variations
        words1 = name1.split()
        words2 = name2.split()
        
        if len(words1) == len(words2) and len(words1) > 1:
            # Check if all words match except one might be plural
            differences = 0
            for w1, w2 in zip(words1, words2):
                if w1 != w2:
                    if not (w1 + 's' == w2 or w2 + 's' == w1 or 
                           w1 + 'es' == w2 or w2 + 'es' == w1):
                        return False
                    differences += 1
            return differences == 1
        
        return False
    
    @staticmethod
    def _are_variants(name1: str, name2: str) -> bool:
        """
        Check if two names are variants of each other (e.g., different formatting).
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            True if they are likely variants
        """
        # Remove all non-alphanumeric characters for comparison
        clean1 = re.sub(r'[^a-z0-9]', '', name1)
        clean2 = re.sub(r'[^a-z0-9]', '', name2)
        
        # Check if they're the same after cleaning
        if clean1 == clean2:
            return True
        
        # Check for model number patterns (letters followed by numbers)
        pattern = r'^([a-z]+)(\d+)$'
        match1 = re.match(pattern, clean1)
        match2 = re.match(pattern, clean2)
        
        if match1 and match2:
            # Same prefix and number
            if match1.group(1) == match2.group(1) and match1.group(2) == match2.group(2):
                return True
        
        return False
    
    @staticmethod
    def _is_compound_term(name1: str, name2: str) -> bool:
        """
        Check if one term is a compound containing the other.
        
        Args:
            name1: First entity name
            name2: Second entity name
            
        Returns:
            True if one contains the other as a meaningful part
        """
        # Normalize for comparison
        norm1 = EntityDeduplicator.normalize_name(name1)
        norm2 = EntityDeduplicator.normalize_name(name2)
        
        # Don't consider single words as compounds
        if ' ' not in norm1 and ' ' not in norm2:
            return False
        
        # Check if one is contained in the other
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # If all words of the shorter term are in the longer term
        if len(words1) < len(words2):
            if words1.issubset(words2):
                logger.debug(f"'{name1}' is a subset of '{name2}'")
                return True
        elif len(words2) < len(words1):
            if words2.issubset(words1):
                logger.debug(f"'{name2}' is a subset of '{name1}'")
                return True
        
        return False
    
    @staticmethod
    def find_duplicates(entities: List[Dict], threshold: float = 0.85, cross_type: bool = True) -> List[List[Dict]]:
        """
        Find duplicate entities based on similarity threshold.
        
        Args:
            entities: List of entity dictionaries with 'name' and 'type' fields
            threshold: Similarity threshold for considering entities as duplicates (0-1)
            cross_type: If True, find duplicates across different entity types
            
        Returns:
            List of duplicate groups, where each group is a list of similar entities
        """
        if not entities:
            return []
        
        duplicate_groups = []
        
        if cross_type:
            # Find duplicates across all types
            grouped_indices = set()
            
            for i, entity1 in enumerate(entities):
                if i in grouped_indices:
                    continue
                
                # Start a new group with this entity
                group = [entity1]
                grouped_indices.add(i)
                
                # Find all similar entities (including different types)
                for j, entity2 in enumerate(entities[i+1:], start=i+1):
                    if j in grouped_indices:
                        continue
                    
                    name1 = entity1.get('entity_name', entity1.get('name', ''))
                    name2 = entity2.get('entity_name', entity2.get('name', ''))
                    
                    similarity = EntityDeduplicator.calculate_similarity(name1, name2)
                    
                    # Check for compound term relationship
                    is_compound = EntityDeduplicator._is_compound_term(name1, name2)
                    
                    if similarity >= threshold or is_compound:
                        group.append(entity2)
                        grouped_indices.add(j)
                        type1 = entity1.get('entity_type', entity1.get('type', 'other'))
                        type2 = entity2.get('entity_type', entity2.get('type', 'other'))
                        logger.info(f"Found cross-type duplicate: '{name1}' ({type1}) ~ '{name2}' ({type2}) (similarity: {similarity:.2f})")
                
                # Only add groups with duplicates
                if len(group) > 1:
                    duplicate_groups.append(group)
        else:
            # Original behavior: Group entities by type first
            entities_by_type = {}
            for entity in entities:
                entity_type = entity.get('entity_type', entity.get('type', 'other')).lower()
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append(entity)
            
            # Find duplicates within each type
            for entity_type, typed_entities in entities_by_type.items():
                # Track which entities have been grouped
                grouped = set()
                
                for i, entity1 in enumerate(typed_entities):
                    if i in grouped:
                        continue
                    
                    # Start a new group with this entity
                    group = [entity1]
                    grouped.add(i)
                    
                    # Find all similar entities
                    for j, entity2 in enumerate(typed_entities[i+1:], start=i+1):
                        if j in grouped:
                            continue
                        
                        name1 = entity1.get('entity_name', entity1.get('name', ''))
                        name2 = entity2.get('entity_name', entity2.get('name', ''))
                        
                        similarity = EntityDeduplicator.calculate_similarity(name1, name2)
                        
                        if similarity >= threshold:
                            group.append(entity2)
                            grouped.add(j)
                            logger.info(f"Found duplicate: '{name1}' ~ '{name2}' (similarity: {similarity:.2f})")
                    
                    # Only add groups with duplicates
                    if len(group) > 1:
                        duplicate_groups.append(group)
        
        return duplicate_groups
    
    @staticmethod
    def merge_entity_data(entities: List[Dict]) -> Dict:
        """
        Merge data from multiple duplicate entities.
        
        Args:
            entities: List of duplicate entities to merge
            
        Returns:
            Merged entity data
        """
        if not entities:
            return {}
        
        # Type priority order (more specific to less specific)
        TYPE_PRIORITY = {
            'component': 1,
            'procedure': 2,
            'problem': 3,
            'specification': 4,
            'system': 5,
            'technology': 6,
            'chemical': 7,
            'product': 8,
            'event': 9,
            'organization': 10,
            'person': 11,
            'location': 12,
            'date': 13,
            'measurement': 14,
            'state': 15,
            'condition': 16,
            'concept': 17,  # Generic types have lower priority
            'other': 18
        }
        
        # Sort entities by type priority first, then confidence
        # This ensures we keep the most specific type
        entities_sorted = sorted(entities, 
                               key=lambda x: (
                                   TYPE_PRIORITY.get(x.get('entity_type', x.get('type', 'other')).lower(), 99),
                                   -x.get('confidence_score', x.get('confidence', 0))
                               ))
        
        merged = entities_sorted[0].copy()
        
        # Collect all unique names
        all_names = set()
        for entity in entities:
            name = entity.get('entity_name', entity.get('name', ''))
            if name:
                all_names.add(name)
        
        # Update merged entity
        merged['confidence_score'] = max(e.get('confidence_score', e.get('confidence', 0)) 
                                        for e in entities)
        
        # Merge metadata
        merged_metadata = merged.get('metadata', {})
        merged_metadata['original_names'] = list(all_names)
        merged_metadata['merge_count'] = len(entities)
        
        # Collect all contexts
        all_contexts = []
        all_chunk_ids = []
        
        for entity in entities:
            metadata = entity.get('metadata', {})
            if metadata.get('context'):
                all_contexts.append(metadata['context'])
            if metadata.get('chunk_id'):
                all_chunk_ids.append(metadata['chunk_id'])
        
        if all_contexts:
            merged_metadata['all_contexts'] = all_contexts
        if all_chunk_ids:
            merged_metadata['chunk_ids'] = list(set(all_chunk_ids))
        
        merged['metadata'] = merged_metadata
        
        # Set normalized name
        merged['normalized_name'] = EntityDeduplicator.normalize_name(
            merged.get('entity_name', merged.get('name', ''))
        )
        
        return merged
    
    @staticmethod
    def deduplicate_entities(entities: List[Dict], 
                           auto_merge_threshold: float = 0.95,
                           review_threshold: float = 0.85,
                           cross_type: bool = True) -> Tuple[List[Dict], List[List[Dict]]]:
        """
        Deduplicate entities with automatic merging for high confidence matches.
        
        Args:
            entities: List of entities to deduplicate
            auto_merge_threshold: Threshold for automatic merging (default 0.95)
            review_threshold: Threshold for flagging for review (default 0.85)
            cross_type: If True, deduplicate across different entity types
            
        Returns:
            Tuple of (deduplicated entities, groups needing review)
        """
        # Find all duplicate groups (with cross-type support)
        all_duplicates = EntityDeduplicator.find_duplicates(entities, review_threshold, cross_type=cross_type)
        
        # Separate into auto-merge and review groups
        auto_merge_groups = []
        review_groups = []
        
        for group in all_duplicates:
            # Calculate average similarity within group
            total_similarity = 0
            count = 0
            
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    name1 = group[i].get('entity_name', group[i].get('name', ''))
                    name2 = group[j].get('entity_name', group[j].get('name', ''))
                    similarity = EntityDeduplicator.calculate_similarity(name1, name2)
                    total_similarity += similarity
                    count += 1
            
            avg_similarity = total_similarity / count if count > 0 else 0
            
            if avg_similarity >= auto_merge_threshold:
                auto_merge_groups.append(group)
            else:
                review_groups.append(group)
        
        # Create set of entities to remove (those being merged)
        entities_to_remove = set()
        merged_entities = []
        
        for group in auto_merge_groups:
            # Mark all entities in group for removal
            for entity in group:
                entity_id = entity.get('id')
                if entity_id:
                    entities_to_remove.add(entity_id)
            
            # Create merged entity
            merged = EntityDeduplicator.merge_entity_data(group)
            merged_entities.append(merged)
            
            logger.info(f"Auto-merged {len(group)} entities into: {merged.get('entity_name', '')}")
        
        # Create final entity list
        deduplicated = []
        
        # Add non-duplicate entities
        for entity in entities:
            entity_id = entity.get('id')
            if entity_id not in entities_to_remove:
                # Check if this entity is in any review group
                in_review_group = False
                for group in review_groups:
                    if any(e.get('id') == entity_id for e in group):
                        in_review_group = True
                        break
                
                if not in_review_group:
                    deduplicated.append(entity)
        
        # Add merged entities
        deduplicated.extend(merged_entities)
        
        # Add entities from review groups (not merged yet)
        for group in review_groups:
            deduplicated.extend(group)
        
        return deduplicated, review_groups