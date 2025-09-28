'use client';

import { useMemo, useState } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { documentApi } from '@/lib/api';
import { useNotification } from '@/components/NotificationProvider';
import GraphPreview from './GraphPreview';

interface Entity {
  id: string;
  entity_name: string;
  entity_type: string;
  confidence: number;
  metadata?: Record<string, any>;
  canonical_metadata?: Record<string, any>;
  properties?: Record<string, any>;
  chunk_ids?: string[];
  is_verified?: boolean;
  is_edited?: boolean;
  original_name?: string;
  canonical_entity_id?: string | null;
  canonical_name?: string | null;
  canonical_type?: string | null;
}

interface CanonicalEntity {
  id: string;
  entity_name: string;
  entity_type: string;
  confidence: number;
  metadata?: Record<string, any>;
  mentions: Entity[];
}

interface Relationship {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relationship_type: string;
  raw_relationship_type?: string;  // Original LLM-generated descriptive type
  confidence: number;
  properties?: Record<string, any>;
  relationship_label?: string;
}

interface EntitiesRelationshipsProps {
  documentId: string;
  entities: Entity[];
}

// Fixed vocabulary of 20 relationship types - DO NOT MODIFY
// These must match the backend RELATIONSHIP_TYPES_CANONICAL in app/utils/relationship_types.py
const RELATIONSHIP_TYPES = [
  // Core structural relationships (6)
  'COMPONENT_OF',     // Part-whole relationships
  'CONNECTED_TO',     // Physical/logical connections
  'DEPENDS_ON',       // Dependencies
  'USES',             // Usage relationships
  'OWNED_BY',         // Ownership
  'RESPONSIBLE_FOR',  // Accountability

  // Process & flow (4)
  'CAUSES',           // Causation
  'PREVENTS',         // Prevention
  'IMPACTS',          // Impact/influence
  'MITIGATES',        // Risk mitigation

  // Knowledge & documentation (4)
  'DEFINES',          // Definitions
  'DESCRIBES',        // Descriptions
  'DOCUMENTS',        // Documentation
  'REFERENCES',       // References/citations

  // State & compatibility (3)
  'REPLACES',         // Replacement
  'COMPATIBLE_WITH',  // Compatibility
  'CONFLICTS_WITH',   // Conflicts

  // Technical operations (2)
  'MONITORS',         // Monitoring
  'MEASURES',         // Measurement

  // Flexible catch-all (1)
  'RELATES_TO',       // For uncategorized relationships
];

const RELATIONSHIP_TYPE_ALIASES: Record<string, string> = {
  'may cause': 'CAUSES',
  'cause': 'CAUSES',
  'causes': 'CAUSES',
  'helps prevent': 'MITIGATES',
  'helps prevent removal': 'MITIGATES',
  'helps prevent removal of': 'MITIGATES',
  'prevents removal of': 'MITIGATES',
  'prevents': 'PREVENTS',
  'reduces': 'MITIGATES',
  'used to clean': 'USES',
  'used for cleaning': 'USES',
  'used for': 'USES',
  'contains': 'COMPONENT_OF',
  'contain': 'COMPONENT_OF',
  'contained in': 'COMPONENT_OF',
  'located in': 'COMPONENT_OF',
  'located_in': 'COMPONENT_OF',
  'emits noise from': 'CAUSES',
  'emits_noise_from': 'CAUSES',
  'noise amplified when': 'IMPACTS',
  'noise_amplified_when': 'IMPACTS',
  'supports': 'RESPONSIBLE_FOR',
  'support': 'RESPONSIBLE_FOR',
};

// Entity type color mapping
const ENTITY_TYPE_COLORS: Record<string, string> = {
  // People & Organizations
  'person': '#8B5CF6',        // Purple
  'organization': '#6366F1',   // Indigo
  
  // Physical & Technical
  'component': '#EF4444',      // Red
  'product': '#F97316',        // Orange
  'system': '#F59E0B',         // Amber
  'technology': '#EAB308',     // Yellow
  'material': '#22C55E',       // Green
  'tool': '#2563EB',           // Blue
  
  // Chemical & Scientific
  'chemical': '#10B981',       // Emerald
  'measurement': '#14B8A6',    // Teal
  'specification': '#06B6D4',  // Cyan
  
  // Procedures & Concepts
  'procedure': '#3B82F6',      // Blue
  'concept': '#8B5CF6',        // Violet
  'event': '#EC4899',          // Pink
  'problem': '#DC2626',        // Red-600 for problems/issues
  
  // States & Conditions
  'state': '#7C3AED',          // Purple-600 for operational states
  'condition': '#A78BFA',      // Purple-400 for physical conditions
  
  // Location & Time
  'location': '#84CC16',       // Lime
  'date': '#737373',           // Gray
  
  // Fallback
  'other': '#9CA3AF',          // Gray
};

// Get entity type color
const getEntityTypeColor = (type: string): string => {
  return ENTITY_TYPE_COLORS[type?.toLowerCase()] || ENTITY_TYPE_COLORS['other'];
};

const KNOWN_ENTITY_TYPES = new Set(Object.keys(ENTITY_TYPE_COLORS));

const normalizeEntityType = (type?: string | null): string | null => {
  if (!type) {
    return null;
  }
  const lower = type.toLowerCase();
  return KNOWN_ENTITY_TYPES.has(lower) ? lower : null;
};

const sanitizeRelationshipType = (type: string): string =>
  type
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'RELATES_TO';

const canonicalizeRelationshipType = (type?: string | null): string => {
  if (!type) {
    return 'RELATES_TO';
  }
  const trimmed = type.trim();
  if (!trimmed) {
    return 'RELATES_TO';
  }
  const lower = trimmed.toLowerCase();
  const alias = RELATIONSHIP_TYPE_ALIASES[lower];
  if (alias) {
    return alias;
  }
  return sanitizeRelationshipType(trimmed);
};

const toTitleCase = (value: string): string =>
  value
    .toLowerCase()
    .split(' ')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

const formatRelationshipTypeLabel = (canonicalType: string): string =>
  toTitleCase(canonicalType.replace(/_/g, ' '));

const getRelationshipDisplayLabel = (
  rawType: string | null | undefined,
  canonicalType: string
): string => {
  // If we have a descriptive raw type that's different from the canonical, use it
  if (rawType && rawType.trim()) {
    const canonicalFromRaw = canonicalizeRelationshipType(rawType);
    // If the raw type maps to a different canonical type, it's a descriptive label
    if (canonicalFromRaw !== rawType) {
      return toTitleCase(rawType.trim().replace(/\s+/g, ' '));
    }
    // Otherwise use the formatted canonical
    return formatRelationshipTypeLabel(canonicalType);
  }
  return formatRelationshipTypeLabel(canonicalType);
};

const formatMetadataValue = (value: any): string => {
  if (value === null || value === undefined) {
    return '';
  }
  if (typeof value === 'string') {
    return value.trim();
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => formatMetadataValue(item))
      .filter((item) => item.length > 0)
      .join(', ');
  }
  if (typeof value === 'object') {
    const inner = Object.entries(value)
      .map(([key, val]) => {
        const formatted = formatMetadataValue(val);
        return formatted.length > 0 ? `${key}: ${formatted}` : '';
      })
      .filter(Boolean)
      .join(', ');
    return inner;
  }
  return String(value);
};

const collectEntityProperties = (entity: Entity): [string, string][] => {
  const canonicalMetadata = entity.canonical_metadata ?? {};
  const mentionMetadata = entity.metadata ?? {};
  const baseProperties = entity.properties ?? {};

  const IGNORED_TOP_LEVEL = new Set([
    'chunk_index',
    'chunking_strategy',
    'chunk_level',
    'chunk_id',
    'document_ids',
    'relationship_document_ids',
    'last_refreshed_at',
    'original_names',
  ]);
  const IGNORED_ATTRIBUTE_KEYS = new Set([
    'chunk_index',
    'chunking_strategy',
    'chunk_level',
    'chunk_id',
    'summary',
  ]);

  const combined: Record<string, any> = {
    ...baseProperties,
    ...mentionMetadata,
  };

  const entries: [string, string][] = [];
  const candidateDescription = [
    canonicalMetadata?.description,
    mentionMetadata?.description,
    baseProperties?.description,
  ].find(
    (value): value is string =>
      typeof value === 'string' && value.trim().length > 0,
  );

  if (candidateDescription) {
    entries.push(['description', formatMetadataValue(candidateDescription)]);
  }

  delete combined.description;

  const detailEntries: [string, string][] = [];

  for (const [key, value] of Object.entries(combined)) {
    if (key === 'context' || key === 'description' || IGNORED_TOP_LEVEL.has(key)) {
      continue;
    }

    if (key === 'attributes' && value && typeof value === 'object') {
      const attributeEntries = Object.entries(value as Record<string, any>)
        .filter(([attrKey]) => !IGNORED_ATTRIBUTE_KEYS.has(attrKey))
        .map(([attrKey, attrValue]) => [attrKey, formatMetadataValue(attrValue)] as [string, string])
        .filter(([, formatted]) => formatted.length > 0);
      detailEntries.push(...attributeEntries);
      continue;
    }

    const formatted = formatMetadataValue(value);
    if (!formatted.length) {
      continue;
    }
    detailEntries.push([key, formatted]);
  }

  entries.push(...detailEntries);

  if (entries.length === 0) {
    const contextValue = formatMetadataValue(
      mentionMetadata?.context ?? canonicalMetadata?.context ?? '',
    );
    if (contextValue.length > 0) {
      entries.push(['context', contextValue]);
    }
  }

  return entries;
};

export default function EntitiesRelationships({ documentId, entities }: EntitiesRelationshipsProps) {
  const queryClient = useQueryClient();
  const { notify, confirm, prompt } = useNotification();
  
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [filterType, setFilterType] = useState<string>('all');
  const [minConfidence, setMinConfidence] = useState(70);
  const [showAddRelationship, setShowAddRelationship] = useState(false);
  const [newRelationship, setNewRelationship] = useState({
    sourceEntity: '',
    targetEntity: '',
    type: 'RELATES_TO',
  });
  const [showAddEntity, setShowAddEntity] = useState(false);
  const [newEntity, setNewEntity] = useState({
    name: '',
    type: 'other',
    confidence: 100,
  });
  const [showMetadataDialog, setShowMetadataDialog] = useState<string | null>(null);
  const [showDuplicatesDialog, setShowDuplicatesDialog] = useState(false);
  const [duplicateGroups, setDuplicateGroups] = useState<any[]>([]);
  const [isSearchingDuplicates, setIsSearchingDuplicates] = useState(false);

  const canonicalEntities = useMemo<CanonicalEntity[]>(() => {
    if (!entities || entities.length === 0) {
      return [];
    }

    const map = new Map<string, CanonicalEntity>();

    for (const entity of entities) {
      const canonicalId = entity.canonical_entity_id || entity.id;
      const displayName = entity.canonical_name || entity.entity_name;
      const resolvedType = normalizeEntityType(entity.canonical_type)
        ?? normalizeEntityType(entity.entity_type)
        ?? 'other';
      const existing = map.get(canonicalId);

      if (existing) {
        existing.confidence = Math.max(existing.confidence, entity.confidence);
        if (!existing.entity_name && displayName) {
          existing.entity_name = displayName;
        }
        if (
          existing.entity_type === 'other' ||
          !normalizeEntityType(existing.entity_type)
        ) {
          const candidateType = normalizeEntityType(entity.entity_type)
            ?? (resolvedType !== 'other' ? resolvedType : null);
          if (candidateType) {
            existing.entity_type = candidateType;
          }
        }
        if (entity.canonical_metadata) {
          existing.metadata = entity.canonical_metadata;
        } else if (!existing.metadata && entity.metadata) {
          existing.metadata = entity.metadata;
        }
        existing.mentions.push(entity);
      } else {
        map.set(canonicalId, {
          id: canonicalId,
          entity_name: displayName || entity.entity_name,
          entity_type: resolvedType,
          confidence: entity.confidence,
          metadata: entity.canonical_metadata ?? entity.metadata,
          mentions: [entity],
        });
      }
    }

    return Array.from(map.values());
  }, [entities]);

  const canonicalEntityMap = useMemo(() => {
    const map = new Map<string, CanonicalEntity>();
    for (const entity of canonicalEntities) {
      map.set(entity.id, entity);
    }
    return map;
  }, [canonicalEntities]);

  const entityMentionMap = useMemo(() => {
    const map = new Map<string, Entity>();
    if (entities) {
      for (const entity of entities) {
        map.set(entity.id, entity);
      }
    }
    return map;
  }, [entities]);

  const resolveCanonicalEntity = (identifier: string | null | undefined): CanonicalEntity | undefined => {
    if (!identifier) {
      return undefined;
    }
    const direct = canonicalEntityMap.get(identifier);
    if (direct) {
      return direct;
    }
    const mention = entityMentionMap.get(identifier);
    if (!mention) {
      return undefined;
    }
    const fallbackId = mention.canonical_entity_id || mention.id;
    return fallbackId ? canonicalEntityMap.get(fallbackId) : undefined;
  };

  // Fetch relationships
  const { data: relationships = [] } = useQuery({
    queryKey: ['relationships', documentId],
    queryFn: async () => {
      try {
        const response = await documentApi.getRelationships(documentId);
        return response.data;
      } catch (error) {
        console.error('Error fetching relationships:', error);
        return [];
      }
    },
  });

  const normalizedRelationships = useMemo(() => {
    if (!relationships.length) {
      return [];
    }
    return relationships.map((rel) => {
      const sourceCanonical = resolveCanonicalEntity(rel.source_entity_id);
      const targetCanonical = resolveCanonicalEntity(rel.target_entity_id);
      const relationshipType = canonicalizeRelationshipType(rel.relationship_type);
      // Use raw_relationship_type if available, otherwise fall back to relationship_type
      const rawType = rel.raw_relationship_type || rel.relationship_type;
      return {
        ...rel,
        source_entity_id: sourceCanonical?.id || rel.source_entity_id,
        target_entity_id: targetCanonical?.id || rel.target_entity_id,
        relationship_type: relationshipType,
        relationship_label: getRelationshipDisplayLabel(rawType, relationshipType),
        raw_relationship_type: rawType,
      };
    });
  }, [relationships, canonicalEntityMap, entityMentionMap]);

  // Use fixed vocabulary - no dynamic additions from data
  const relationshipTypeOptions = useMemo(() => {
    // Return the fixed list, sorted for consistent UI
    return [...RELATIONSHIP_TYPES].sort();
  }, []);

  // Update entity mutation
  const updateEntityMutation = useMutation({
    mutationFn: ({ entityId, data }: { entityId: string; data: any }) =>
      documentApi.updateEntity(entityId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
      notify('Entity updated successfully', 'success');
    },
    onError: (error) => {
      notify(`Failed to update entity: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  // Delete entity mutation
  const deleteEntityMutation = useMutation({
    mutationFn: documentApi.deleteEntity,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
      notify('Entity deleted successfully', 'success');
    },
    onError: (error) => {
      notify(`Failed to delete entity: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  // Create entity mutation
  const createEntityMutation = useMutation({
    mutationFn: (data: any) => documentApi.createEntity(documentId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
      queryClient.invalidateQueries({ queryKey: ['document', documentId] });
      notify('Entity created successfully', 'success');
      setShowAddEntity(false);
      setNewEntity({ name: '', type: 'other', confidence: 100 });
    },
    onError: (error: any) => {
      notify(`Failed to create entity: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  // Create relationship mutation
  const createRelationshipMutation = useMutation({
    mutationFn: (data: any) => documentApi.createRelationship(documentId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['relationships', documentId] });
      notify('Relationship created successfully', 'success');
      setShowAddRelationship(false);
      setNewRelationship({ sourceEntity: '', targetEntity: '', type: 'RELATES_TO' });
    },
    onError: (error: any) => {
      notify(`Failed to create relationship: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  // Approve relationship mutation
  const approveRelationshipMutation = useMutation({
    mutationFn: (relationshipId: string) => 
      documentApi.updateRelationship(relationshipId, { approved: true }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['relationships', documentId] });
      notify('Relationship approved', 'success');
    },
  });

  // Update relationship mutation
  const updateRelationshipMutation = useMutation({
    mutationFn: ({ id, ...updates }: { id: string; relationship_type?: string; source_entity_id?: string; target_entity_id?: string }) => 
      documentApi.updateRelationship(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['relationships', documentId] });
      notify('Relationship updated', 'success');
    },
    onError: (error: any) => {
      notify('Failed to update relationship: ' + (error.message || 'Unknown error'), 'error');
    },
  });

  // Delete relationship mutation
  const deleteRelationshipMutation = useMutation({
    mutationFn: (relationshipId: string) => 
      documentApi.deleteRelationship(relationshipId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['relationships', documentId] });
      notify('Relationship deleted', 'success');
    },
    onError: (error: any) => {
      notify('Failed to delete relationship: ' + (error.message || 'Unknown error'), 'error');
    },
  });

  // Find duplicates function
  const findDuplicates = async (autoMerge: boolean = false) => {
    setIsSearchingDuplicates(true);
    try {
      const response = await documentApi.findDuplicates(documentId, 0.85, autoMerge);
      
      if (autoMerge && response.data.merged > 0) {
        notify(`Auto-merged ${response.data.merged} duplicate entities`, 'success');
        queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
        queryClient.invalidateQueries({ queryKey: ['document', documentId] });
      }
      
      if (response.data.duplicates.length > 0) {
        setDuplicateGroups(response.data.duplicates);
        setShowDuplicatesDialog(true);
      } else {
        notify('No duplicate entities found', 'info');
      }
    } catch (error: any) {
      notify('Failed to find duplicates: ' + (error.message || 'Unknown error'), 'error');
    } finally {
      setIsSearchingDuplicates(false);
    }
  };

  // Merge duplicate group
  const mergeDuplicateGroup = async (group: any) => {
    try {
      const entityIds = group.entities.map((e: any) => e.id);
      const targetName = group.entities[0].entity_name;
      const targetType = group.entities[0].entity_type;
      
      await documentApi.mergeEntities(entityIds, targetName, targetType);
      
      notify(`Merged ${entityIds.length} entities`, 'success');
      
      // Refresh data
      queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
      queryClient.invalidateQueries({ queryKey: ['document', documentId] });
      
      // Remove merged group from list
      setDuplicateGroups(prev => prev.filter(g => g !== group));
      
      if (duplicateGroups.length <= 1) {
        setShowDuplicatesDialog(false);
      }
    } catch (error: any) {
      notify('Failed to merge entities: ' + (error.message || 'Unknown error'), 'error');
    }
  };

  // Filter entities
  // Get unique entity types for filter (canonical view)
  const entityTypes = useMemo(
    () => [...new Set(canonicalEntities.map((entity) => entity.entity_type).filter(Boolean))],
    [canonicalEntities]
  );

  // Get entity mention by ID
  const getMentionById = (id: string) => entityMentionMap.get(id);

  const filteredCanonicalEntities = useMemo(() => {
    if (!canonicalEntities.length) {
      return [];
    }
    return canonicalEntities.filter((entity) => {
      if (filterType !== 'all' && entity.entity_type !== filterType) {
        return false;
      }
      if (entity.confidence * 100 < minConfidence) {
        return false;
      }
      return true;
    });
  }, [canonicalEntities, filterType, minConfidence]);

  const graphEntities = useMemo(() =>
    filteredCanonicalEntities.map((entity) => ({
      id: entity.id,
      entity_name: entity.entity_name,
      entity_type: entity.entity_type,
      confidence: entity.confidence,
      metadata: entity.metadata ?? {},
    })),
    [filteredCanonicalEntities]
  );

  return (
    <div className="space-y-6">
      {/* Entities Section */}
      <div>
        <div style={{ border: '1px solid #E1E8ED', marginBottom: '24px' }}>
          <div style={{ padding: '20px', borderBottom: '1px solid #E1E8ED' }}>
            <div className="flex justify-between items-center">
              <h3 className="font-bold" style={{ color: '#2C3E50' }}>Extracted Entities</h3>
              <div className="flex items-center gap-4">
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <label className="text-sm" style={{ color: '#2C3E50', fontWeight: '500', marginRight: '6px' }}>Filter:</label>
                  <select
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    style={{
                      border: '1px solid #E1E8ED',
                      borderRadius: '6px',
                      padding: '8px 12px',
                      paddingRight: '36px',
                      fontSize: '14px',
                      backgroundColor: 'white',
                      minWidth: '140px'
                    }}
                  >
                    <option value="all">All Types</option>
                    {entityTypes.map(type => (
                      <option key={type} value={type}>{type}</option>
                    ))}
                  </select>
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <label className="text-sm" style={{ color: '#2C3E50', fontWeight: '500', marginRight: '6px' }}>Confidence:</label>
                  <select
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(parseInt(e.target.value))}
                    style={{
                      border: '1px solid #E1E8ED',
                      borderRadius: '6px',
                      padding: '8px 12px',
                      paddingRight: '36px',
                      fontSize: '14px',
                      backgroundColor: 'white',
                      minWidth: '100px'
                    }}
                  >
                    <option value={0}>All</option>
                    <option value={50}>&gt;50%</option>
                    <option value={70}>&gt;70%</option>
                    <option value={90}>&gt;90%</option>
                  </select>
                </div>
                <button
                  onClick={() => findDuplicates(false)}
                  disabled={isSearchingDuplicates || !entities || entities.length < 2}
                  style={{
                    backgroundColor: 'white',
                    color: '#9B59B6',
                    border: '1px solid #D4B5E0',
                    padding: '8px 16px',
                    fontSize: '14px',
                    fontWeight: '500',
                    marginLeft: '12px',
                    opacity: isSearchingDuplicates || !entities || entities.length < 2 ? 0.5 : 1,
                    cursor: isSearchingDuplicates || !entities || entities.length < 2 ? 'not-allowed' : 'pointer'
                  }}
                  className="hover:bg-purple-50"
                  title="Find duplicate entities"
                >
                  {isSearchingDuplicates ? 'Searching...' : '🔍 Find Duplicates'}
                </button>
                <button
                  onClick={() => setShowAddEntity(!showAddEntity)}
                  style={{
                    backgroundColor: 'white',
                    color: '#3498DB',
                    border: '1px solid #BDD7ED',
                    padding: '8px 16px',
                    fontSize: '14px',
                    fontWeight: '500',
                    marginLeft: '8px'
                  }}
                  className="hover:bg-blue-50"
                >
                  {showAddEntity ? 'Cancel' : '+ Add Entity'}
                </button>
              </div>
            </div>
          </div>
          
          {showAddEntity && (
            <div style={{ padding: '20px', borderBottom: '1px solid #E1E8ED', backgroundColor: '#F8FBFF' }}>
              <div className="flex items-center gap-4">
                <select
                  value={newEntity.type}
                  onChange={(e) => setNewEntity({...newEntity, type: e.target.value})}
                  style={{
                    border: '1px solid #E1E8ED',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    paddingRight: '36px',
                    fontSize: '14px',
                    backgroundColor: 'white',
                    minWidth: '140px'
                  }}
                >
                  <option value="person">Person</option>
                  <option value="organization">Organization</option>
                  <option value="location">Location</option>
                  <option value="date">Date</option>
                  <option value="product">Product</option>
                  <option value="component">Component</option>
                  <option value="technology">Technology</option>
                  <option value="chemical">Chemical</option>
                  <option value="procedure">Procedure</option>
                  <option value="specification">Specification</option>
                  <option value="system">System</option>
                  <option value="measurement">Measurement</option>
                  <option value="problem">Problem</option>
                  <option value="concept">Concept</option>
                  <option value="event">Event</option>
                  <option value="other">Other</option>
                </select>
                
                <input
                  type="text"
                  value={newEntity.name}
                  onChange={(e) => setNewEntity({...newEntity, name: e.target.value})}
                  placeholder="Entity name..."
                  style={{
                    border: '1px solid #E1E8ED',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    fontSize: '14px',
                    backgroundColor: 'white',
                    minWidth: '200px'
                  }}
                />
                
                <input
                  type="text"
                  placeholder="Context (optional)..."
                  style={{
                    border: '1px solid #E1E8ED',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    fontSize: '14px',
                    backgroundColor: 'white',
                    minWidth: '300px',
                    flex: 1
                  }}
                />
                
                <div className="flex items-center gap-2">
                  <label className="text-sm" style={{ color: '#2C3E50' }}>Confidence:</label>
                  <input
                    type="number"
                    value={newEntity.confidence}
                    onChange={(e) => setNewEntity({...newEntity, confidence: parseInt(e.target.value) || 100})}
                    min="0"
                    max="100"
                    style={{
                      border: '1px solid #E1E8ED',
                      borderRadius: '6px',
                      padding: '8px 12px',
                      fontSize: '14px',
                      backgroundColor: 'white',
                      width: '90px'
                    }}
                  />
                  <span className="text-sm" style={{ color: '#7F8C8D' }}>%</span>
                </div>
                
                <button
                  onClick={() => {
                    if (newEntity.name.trim()) {
                      createEntityMutation.mutate({
                        entity_name: newEntity.name.trim(),
                        entity_type: newEntity.type,
                        confidence: newEntity.confidence / 100,
                        manual: true,
                      });
                    } else {
                      notify('Please enter an entity name', 'warning');
                    }
                  }}
                  style={{
                    backgroundColor: '#3498DB',
                    color: 'white',
                    padding: '6px 16px',
                    fontSize: '14px',
                    fontWeight: '500',
                    border: 'none'
                  }}
                  className="hover:opacity-90"
                >
                  Create
                </button>
                
                <button
                  onClick={() => {
                    setShowAddEntity(false);
                    setNewEntity({ name: '', type: 'other', confidence: 100 });
                  }}
                  style={{
                    color: '#E74C3C',
                    fontSize: '20px',
                    width: '32px',
                    height: '32px',
                    padding: '0',
                    background: 'none',
                    border: '1px solid #E74C3C30',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginLeft: 'auto'
                  }}
                  title="Cancel"
                >
                  ×
                </button>
              </div>
            </div>
          )}
          
          {/* Entities Table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ 
              width: '100%', 
              borderCollapse: 'collapse',
              fontSize: '14px'
            }}>
              <thead>
                <tr style={{ 
                  backgroundColor: '#F8F7F3',
                  borderBottom: '2px solid #E1E8ED'
                }}>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '10%'
                  }}>
                    Type
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '15%'
                  }}>
                    Entity
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50'
                  }}>
                    Properties
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '8%',
                    minWidth: '85px'
                  }}>
                    Confidence
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '110px'  // Fixed width: 3 buttons * 32px + 2 gaps * 4px + minimal padding
                  }}>
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredCanonicalEntities.map((canonical) => {
                  const representative = canonical.mentions[0];
                  const displayEntity: Entity = representative
                    ? {
                        ...representative,
                        entity_name: canonical.entity_name,
                        entity_type: canonical.entity_type,
                        confidence: canonical.confidence,
                        canonical_metadata: canonical.metadata ?? {},
                        canonical_entity_id: canonical.id,
                      }
                    : {
                        id: canonical.id,
                        entity_name: canonical.entity_name,
                        entity_type: canonical.entity_type,
                        confidence: canonical.confidence,
                        metadata: canonical.metadata ?? {},
                        canonical_metadata: canonical.metadata ?? {},
                        canonical_entity_id: canonical.id,
                      } as Entity;

                  return (
                  <tr 
                    key={canonical.id} 
                    style={{ 
                      borderBottom: '1px solid #F0EDE5',
                      transition: 'background-color 0.2s',
                      cursor: 'pointer'
                    }}
                    className="hover:bg-gray-50"
                    onClick={() => representative && setSelectedEntity(representative)}
                  >
                    <td style={{ padding: '12px' }}>
                      <span 
                        style={{ 
                          backgroundColor: getEntityTypeColor(canonical.entity_type),
                          color: 'white',
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '600',
                          textTransform: 'uppercase'
                        }}
                      >
                        {canonical.entity_type}
                      </span>
                    </td>
                    <td style={{ padding: '12px', fontWeight: '500', color: '#2C3E50' }}>
                      {canonical.entity_name}
                    </td>
                    <td style={{ 
                      padding: '12px', 
                      color: '#7F8C8D',
                      fontSize: '13px'
                    }}>
                      {(() => {
                        const propertyEntries = collectEntityProperties(displayEntity).slice(0, 3);
                        if (propertyEntries.length === 0) {
                          return <span style={{ opacity: 0.5 }}>No properties available</span>;
                        }
                        return (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            {propertyEntries.map(([key, value], index) => (
                              <div
                                key={`${key}-${index}`}
                                style={{
                                  whiteSpace: 'normal',
                                  overflow: 'visible',
                                  textOverflow: 'clip'
                                }}
                                title={`${key}: ${value}`}
                              >
                                <span style={{ fontWeight: 600, color: '#2C3E50' }}>{key}:</span>{' '}
                                <span>{value}</span>
                              </div>
                            ))}
                          </div>
                        );
                      })()}
                    </td>
                    <td style={{ 
                      padding: '12px',
                      textAlign: 'left',
                      color: '#7F8C8D'
                    }}>
                      {(canonical.confidence * 100).toFixed(0)}%
                    </td>
                    <td style={{ 
                      padding: '12px',
                      textAlign: 'right'
                    }}>
                      <div style={{ display: 'flex', gap: '4px', justifyContent: 'flex-end' }}>
                        <button
                          onClick={(e) => {
                           e.stopPropagation();
                            setShowMetadataDialog(representative?.id ?? canonical.id);
                          }}
                          style={{ 
                            color: '#3498DB', 
                            fontSize: '18px', 
                            width: '32px',
                            height: '32px',
                            padding: '0',
                            backgroundColor: '#3498DB10',
                            borderRadius: '4px',
                            border: '1px solid #3498DB30',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}
                          className="hover:bg-blue-100"
                          title="View metadata"
                        >
                          ℹ
                        </button>
                        <button
                          onClick={async (e) => {
                            e.stopPropagation();
                            if (!representative) {
                              notify('Cannot edit canonical entity without a source mention', 'warning');
                              return;
                            }
                            const newName = await prompt(
                              `Edit entity "${canonical.entity_name}"\nType: ${canonical.entity_type}`,
                              canonical.entity_name
                            );
                            if (newName && newName !== canonical.entity_name) {
                              updateEntityMutation.mutate({
                                entityId: representative.id,
                                data: { entity_name: newName }
                              });
                            }
                          }}
                          style={{ 
                            color: '#27AE60', 
                            fontSize: '18px', 
                            width: '32px',
                            height: '32px',
                            padding: '0',
                            backgroundColor: '#27AE6010',
                            borderRadius: '4px',
                            border: '1px solid #27AE6030',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}
                          className="hover:bg-green-100"
                          title="Edit entity"
                        >
                          ✎
                        </button>
                        <button
                          onClick={async (e) => {
                            e.stopPropagation();
                            if (!representative) {
                              notify('Cannot delete canonical entity without a source mention', 'warning');
                              return;
                            }
                            const confirmed = await confirm(
                              `Delete entity "${canonical.entity_name}"?\nType: ${canonical.entity_type}`
                            );
                            if (confirmed) {
                              deleteEntityMutation.mutate(representative.id);
                            }
                          }}
                          style={{ 
                            color: '#E74C3C', 
                            fontSize: '18px', 
                            width: '32px',
                            height: '32px',
                            padding: '0',
                            backgroundColor: '#E74C3C10',
                            borderRadius: '4px',
                            border: '1px solid #E74C3C30',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                          }}
                          className="hover:bg-red-100"
                          title="Delete entity"
                        >
                          ×
                        </button>
                      </div>
                    </td>
                  </tr>
                  );
                })}
             </tbody>
           </table>
           
            {filteredCanonicalEntities.length === 0 && (
              <div style={{
                padding: '32px',
                textAlign: 'center',
                color: '#7F8C8D',
                backgroundColor: '#F8F7F3'
              }}>
                No entities found matching the filters
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Relationships Section */}
      <div>
        <div style={{ border: '1px solid #E1E8ED', marginBottom: '24px' }}>
          <div style={{ padding: '20px', borderBottom: '1px solid #E1E8ED' }}>
            <div className="flex justify-between items-center">
              <h3 className="font-bold" style={{ color: '#2C3E50' }}>Proposed Relationships</h3>
              <button
                onClick={() => setShowAddRelationship(!showAddRelationship)}
                style={{
                  backgroundColor: 'white',
                  color: '#3498DB',
                  border: '1px solid #BDD7ED',
                  padding: '8px 16px',
                  fontSize: '14px',
                  fontWeight: '500'
                }}
                className="hover:bg-blue-50"
              >
                {showAddRelationship ? 'Cancel' : '+ Add Custom Relationship'}
              </button>
            </div>
          </div>
          
          {showAddRelationship && (
            <div style={{ padding: '20px', borderBottom: '1px solid #E1E8ED', backgroundColor: '#F8FBFF' }}>
              <div className="flex items-center gap-4">
                <select
                  value={newRelationship.sourceEntity}
                  onChange={(e) => setNewRelationship({...newRelationship, sourceEntity: e.target.value})}
                  style={{
                    border: '1px solid #E1E8ED',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    paddingRight: '36px',
                    fontSize: '14px',
                    backgroundColor: 'white',
                    minWidth: '200px'
                  }}
                >
                  <option value="">Select source entity...</option>
                  {canonicalEntities.map(entity => (
                    <option key={entity.id} value={entity.id}>
                      {entity.entity_name} ({entity.entity_type})
                    </option>
                  ))}
                </select>
                
                <select
                  value={newRelationship.type}
                  onChange={(e) => setNewRelationship({...newRelationship, type: e.target.value})}
                  style={{
                    border: '1px solid #E1E8ED',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    paddingRight: '36px',
                    fontSize: '14px',
                    backgroundColor: 'white',
                    minWidth: '160px'
                  }}
                >
                  {relationshipTypeOptions.map(type => (
                    <option key={type} value={type}>{formatRelationshipTypeLabel(type)}</option>
                  ))}
                </select>
                
                <select
                  value={newRelationship.targetEntity}
                  onChange={(e) => setNewRelationship({...newRelationship, targetEntity: e.target.value})}
                  style={{
                    border: '1px solid #E1E8ED',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    paddingRight: '36px',
                    fontSize: '14px',
                    backgroundColor: 'white',
                    minWidth: '200px'
                  }}
                >
                  <option value="">Select target entity...</option>
                  {canonicalEntities.map(entity => (
                    <option key={entity.id} value={entity.id}>
                      {entity.entity_name} ({entity.entity_type})
                    </option>
                  ))}
                </select>
                
                <button
                  onClick={() => {
                    if (newRelationship.sourceEntity && newRelationship.targetEntity) {
                      createRelationshipMutation.mutate({
                        source_entity_id: newRelationship.sourceEntity,
                        target_entity_id: newRelationship.targetEntity,
                        relationship_type: newRelationship.type,
                        confidence_score: 1.0,
                        metadata: { manual: true },
                      });
                    } else {
                      notify('Please select both source and target entities', 'warning');
                    }
                  }}
                  style={{
                    backgroundColor: '#3498DB',
                    color: 'white',
                    padding: '6px 16px',
                    fontSize: '14px',
                    fontWeight: '500',
                    border: 'none'
                  }}
                  className="hover:opacity-90"
                >
                  Create
                </button>
              </div>
            </div>
          )}
          
          {/* Relationships Table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ 
              width: '100%', 
              borderCollapse: 'collapse',
              fontSize: '14px'
            }}>
              <thead>
                <tr style={{ 
                  backgroundColor: '#F8F7F3',
                  borderBottom: '2px solid #E1E8ED'
                }}>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '25%'
                  }}>
                    Source
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '20%'
                  }}>
                    Relationship Type
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50'
                  }}>
                    Target
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '8%',
                    minWidth: '85px'
                  }}>
                    Confidence
                  </th>
                  <th style={{ 
                    padding: '12px', 
                    textAlign: 'left', 
                    fontWeight: '600',
                    color: '#2C3E50',
                    width: '149px'
                  }}>
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {relationships?.map((rel: Relationship) => {
                  const sourceCanonical = resolveCanonicalEntity(rel.source_entity_id);
                  const targetCanonical = resolveCanonicalEntity(rel.target_entity_id);
                  const normalizedSourceId = sourceCanonical?.id || rel.source_entity_id || '';
                  const normalizedTargetId = targetCanonical?.id || rel.target_entity_id || '';

                  return (
                    <tr key={rel.id} style={{ 
                      borderBottom: '1px solid #F0EDE5',
                      transition: 'background-color 0.2s'
                    }}
                    className="hover:bg-gray-50"
                    >
                      <td style={{ padding: '12px' }}>
                        <select
                          value={normalizedSourceId}
                          onChange={(e) => updateRelationshipMutation.mutate({
                            id: rel.id,
                            source_entity_id: e.target.value
                          })}
                          style={{
                            width: '100%',
                            padding: '8px 12px',
                            paddingRight: '36px',
                            border: '1px solid #E1E8ED',
                            borderRadius: '4px',
                            backgroundColor: 'white',
                            color: '#2C3E50',
                            fontSize: '13px',
                            cursor: 'pointer'
                          }}
                          className="hover:border-blue-400 focus:outline-none focus:border-blue-500"
                        >
                          {normalizedSourceId && !canonicalEntityMap.has(normalizedSourceId) && (
                            <option value={normalizedSourceId}>
                              {sourceCanonical
                                ? `${sourceCanonical.entity_name} (${sourceCanonical.entity_type})`
                                : `Unknown entity (${normalizedSourceId})`}
                            </option>
                          )}
                          {canonicalEntities.map(entity => (
                            <option key={entity.id} value={entity.id}>
                              {entity.entity_name} ({entity.entity_type})
                            </option>
                          ))}
                        </select>
                      </td>
                      <td style={{ padding: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <select
                            value={rel.relationship_type}
                            onChange={(e) => updateRelationshipMutation.mutate({
                              id: rel.id,
                              relationship_type: e.target.value
                            })}
                            style={{
                              flex: 1,
                              padding: '8px 12px',
                              paddingRight: '36px',
                              border: '1px solid #E1E8ED',
                              borderRadius: '4px',
                              backgroundColor: 'white',
                              color: '#2C3E50',
                              fontSize: '13px',
                              cursor: 'pointer'
                            }}
                            className="hover:border-blue-400 focus:outline-none focus:border-blue-500"
                          >
                            {/* Show the current relationship with its display label */}
                            <option value={rel.relationship_type}>
                              {rel.relationship_label || formatRelationshipTypeLabel(rel.relationship_type)}
                            </option>
                            {/* Show other available options */}
                            {relationshipTypeOptions
                              .filter(type => type !== rel.relationship_type)
                              .map(type => (
                                <option key={type} value={type}>
                                  {formatRelationshipTypeLabel(type)}
                                </option>
                              ))}
                          </select>
                          {/* Show indicator if using descriptive (non-canonical) label */}
                          {rel.raw_relationship_type &&
                           rel.raw_relationship_type !== rel.relationship_type && (
                            <span
                              title={`LLM suggested: "${rel.raw_relationship_type}" → Canonical: "${rel.relationship_type}"`}
                              style={{
                                color: '#6B7280',
                                fontSize: '11px',
                                cursor: 'help',
                                padding: '2px 4px',
                                backgroundColor: '#F3F4F6',
                                borderRadius: '3px',
                                border: '1px solid #E5E7EB',
                                whiteSpace: 'nowrap'
                              }}
                            >
                              ℹ️
                            </span>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '12px' }}>
                        <select
                          value={normalizedTargetId}
                          onChange={(e) => updateRelationshipMutation.mutate({
                            id: rel.id,
                            target_entity_id: e.target.value
                          })}
                          style={{
                            width: '100%',
                            padding: '8px 12px',
                            paddingRight: '36px',
                            border: '1px solid #E1E8ED',
                            borderRadius: '4px',
                            backgroundColor: 'white',
                            color: '#2C3E50',
                            fontSize: '13px',
                            cursor: 'pointer'
                          }}
                          className="hover:border-blue-400 focus:outline-none focus:border-blue-500"
                        >
                          {normalizedTargetId && !canonicalEntityMap.has(normalizedTargetId) && (
                            <option value={normalizedTargetId}>
                              {targetCanonical
                                ? `${targetCanonical.entity_name} (${targetCanonical.entity_type})`
                                : `Unknown entity (${normalizedTargetId})`}
                            </option>
                          )}
                          {canonicalEntities.map(entity => (
                            <option key={entity.id} value={entity.id}>
                              {entity.entity_name} ({entity.entity_type})
                            </option>
                          ))}
                        </select>
                      </td>
                      <td style={{ 
                        padding: '12px',
                        textAlign: 'left',
                        color: '#7F8C8D'
                      }}>
                        {(rel.confidence * 100).toFixed(0)}%
                      </td>
                      <td style={{ 
                        padding: '12px',
                        textAlign: 'right'
                      }}>
                        <div style={{ display: 'flex', gap: '4px', justifyContent: 'flex-end' }}>
                          <button
                            onClick={() => deleteRelationshipMutation.mutate(rel.id)}
                            style={{ 
                              color: '#E74C3C', 
                              fontSize: '18px', 
                              width: '32px',
                              height: '32px',
                              padding: '0',
                              backgroundColor: '#E74C3C10',
                              borderRadius: '4px',
                              border: '1px solid #E74C3C30',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center'
                            }}
                            className="hover:bg-red-100"
                            title="Delete relationship"
                          >
                            ×
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            
            {relationships?.length === 0 && (
              <div style={{
                padding: '32px',
                textAlign: 'center',
                color: '#7F8C8D',
                backgroundColor: '#F8F7F3'
              }}>
                No relationships found for this document
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Graph Preview */}
      <div style={{ border: '1px solid #E1E8ED', padding: '20px' }}>
        <h3 className="font-bold mb-4" style={{ color: '#2C3E50' }}>Graph Preview</h3>
        <GraphPreview 
          entities={graphEntities} 
          relationships={normalizedRelationships} 
        />
      </div>

      {/* Metadata Dialog */}
      {showMetadataDialog && (() => {
        const entity = showMetadataDialog ? getMentionById(showMetadataDialog) : undefined;
        if (!entity) return null;
        
        return (
          <div 
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000
            }}
            onClick={() => setShowMetadataDialog(null)}
          >
            <div
              style={{
                backgroundColor: 'white',
                borderRadius: '8px',
                padding: '24px',
                maxWidth: '600px',
                maxHeight: '80vh',
                overflow: 'auto',
                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-bold" style={{ color: '#2C3E50' }}>
                  Entity Metadata: {entity.entity_name}
                </h3>
                <button
                  onClick={() => setShowMetadataDialog(null)}
                  style={{
                    fontSize: '24px',
                    color: '#7F8C8D',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    padding: '0',
                    lineHeight: '1'
                  }}
                >
                  ×
                </button>
              </div>
              
              <div className="space-y-4">
                {/* Basic Info */}
                <div>
                  <h4 className="font-semibold text-sm mb-2" style={{ color: '#34495E' }}>Basic Information</h4>
                  <div className="space-y-1 text-sm">
                    <div><strong>Type:</strong> {entity.entity_type}</div>
                    <div><strong>Confidence:</strong> {(entity.confidence * 100).toFixed(0)}%</div>
                    {entity.is_verified && <div><strong>Status:</strong> ✓ Verified</div>}
                    {entity.is_edited && entity.original_name && (
                      <div><strong>Original Name:</strong> {entity.original_name}</div>
                    )}
                  </div>
                </div>
                
                {/* Metadata */}
                {entity.metadata && Object.keys(entity.metadata).length > 0 && (
                  <div>
                    <h4 className="font-semibold text-sm mb-2" style={{ color: '#34495E' }}>Extracted Metadata</h4>
                    <div className="space-y-2 text-sm">
                      {entity.metadata.context && (
                        <div>
                          <strong>Context:</strong>
                          <div className="mt-1 p-2 bg-gray-50 rounded" style={{ fontSize: '13px' }}>
                            {entity.metadata.context}
                          </div>
                        </div>
                      )}
                      {entity.metadata.category && (
                        <div><strong>Category:</strong> {entity.metadata.category}</div>
                      )}
                      {entity.metadata.chunk_id && (
                        <div><strong>Source Chunk ID:</strong> {entity.metadata.chunk_id}</div>
                      )}
                      {entity.metadata.attributes && Object.keys(entity.metadata.attributes).length > 0 && (
                        <div>
                          <strong>Type-Specific Attributes:</strong>
                          <div className="mt-1 ml-4 space-y-1">
                            {Object.entries(entity.metadata.attributes).map(([key, value]) => (
                              <div key={key}>
                                • {key}: {String(value)}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Properties (legacy field) */}
                {entity.properties && Object.keys(entity.properties).length > 0 && (
                  <div>
                    <h4 className="font-semibold text-sm mb-2" style={{ color: '#34495E' }}>Additional Properties</h4>
                    <div className="space-y-1 text-sm">
                      {Object.entries(entity.properties).map(([key, value]) => (
                        <div key={key}>
                          <strong>{key}:</strong> {String(value)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Chunk IDs */}
                {entity.chunk_ids && entity.chunk_ids.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-sm mb-2" style={{ color: '#34495E' }}>Found in Chunks</h4>
                    <div className="text-sm">
                      {entity.chunk_ids.join(', ')}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })()}
      
      {/* Duplicates Dialog */}
      {showDuplicatesDialog && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '24px',
            maxWidth: '800px',
            maxHeight: '80vh',
            overflow: 'auto',
            width: '90%'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '20px'
            }}>
              <h3 style={{ fontSize: '20px', fontWeight: '600', color: '#2C3E50' }}>
                Duplicate Entities Found ({duplicateGroups.length} groups)
              </h3>
              <button
                onClick={() => setShowDuplicatesDialog(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '24px',
                  cursor: 'pointer',
                  color: '#7F8C8D'
                }}
              >
                ×
              </button>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <button
                onClick={() => findDuplicates(true)}
                style={{
                  backgroundColor: '#27AE60',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '8px 16px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  marginRight: '8px'
                }}
                className="hover:bg-green-600"
              >
                Auto-Merge All (≥95% Similar)
              </button>
              <span style={{ color: '#7F8C8D', fontSize: '14px' }}>
                Only very similar entities will be merged automatically
              </span>
            </div>

            {duplicateGroups.map((group, index) => (
              <div key={index} style={{
                border: '1px solid #E1E8ED',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '16px',
                backgroundColor: '#FAFBFC'
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '12px'
                }}>
                  <div>
                    <span style={{
                      backgroundColor: group.similarity >= 0.95 ? '#D4EFDF' : 
                                       group.similarity >= 0.9 ? '#FCF3CF' : '#FADBD8',
                      color: group.similarity >= 0.95 ? '#27AE60' : 
                             group.similarity >= 0.9 ? '#F39C12' : '#E74C3C',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}>
                      {(group.similarity * 100).toFixed(0)}% Similar
                    </span>
                    <span style={{
                      marginLeft: '8px',
                      color: '#7F8C8D',
                      fontSize: '12px'
                    }}>
                      {group.entities.length} entities
                    </span>
                  </div>
                  <button
                    onClick={() => mergeDuplicateGroup(group)}
                    style={{
                      backgroundColor: '#3498DB',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '6px 12px',
                      fontSize: '13px',
                      fontWeight: '500',
                      cursor: 'pointer'
                    }}
                    className="hover:bg-blue-600"
                  >
                    Merge Group
                  </button>
                </div>
                
                <div style={{ fontSize: '14px' }}>
                  {group.names.map((name: string, i: number) => (
                    <div key={i} style={{
                      padding: '4px 0',
                      color: '#2C3E50'
                    }}>
                      • {name}
                      {group.entities[i].entity_type && (
                        <span style={{
                          marginLeft: '8px',
                          color: '#7F8C8D',
                          fontSize: '12px'
                        }}>
                          ({group.entities[i].entity_type})
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {duplicateGroups.length === 0 && (
              <div style={{
                textAlign: 'center',
                padding: '40px',
                color: '#7F8C8D'
              }}>
                No duplicates remaining
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
