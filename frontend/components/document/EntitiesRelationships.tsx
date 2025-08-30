'use client';

import { useState } from 'react';
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
  properties?: Record<string, any>;
  chunk_ids?: string[];
  is_verified?: boolean;
  is_edited?: boolean;
  original_name?: string;
}

interface Relationship {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relationship_type: string;
  confidence: number;
  properties?: Record<string, any>;
}

interface EntitiesRelationshipsProps {
  documentId: string;
  entities: Entity[];
}

const RELATIONSHIP_TYPES = [
  // Technical
  'COMPONENT_OF',
  'CONNECTS_TO',
  'DEPENDS_ON',
  'REPLACES',
  'COMPATIBLE_WITH',
  'TROUBLESHOOTS',
  'CAUSES',
  'PREVENTS',
  'REQUIRES',
  'RESOLVES',
  'INDICATES',
  // Documentation
  'DEFINES',
  'DOCUMENTS',
  'REFERENCES',
  'TARGETS',
  // Business
  'RESPONSIBLE_FOR',
  'SERVES',
  'IMPACTS',
  // Flexible
  'RELATES_TO',
];

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

  // Filter entities
  const filteredEntities = entities?.filter(entity => {
    if (filterType !== 'all' && entity.entity_type !== filterType) {
      return false;
    }
    if (entity.confidence * 100 < minConfidence) {
      return false;
    }
    return true;
  });

  // Get unique entity types for filter
  const entityTypes = [...new Set(entities?.map(e => e.entity_type) || [])];

  // Get entity by ID
  const getEntityById = (id: string) => entities?.find(e => e.id === id);

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
                  onClick={() => setShowAddEntity(!showAddEntity)}
                  style={{
                    backgroundColor: 'white',
                    color: '#3498DB',
                    border: '1px solid #BDD7ED',
                    padding: '8px 16px',
                    fontSize: '14px',
                    fontWeight: '500',
                    marginLeft: '12px'
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
                    Context
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
                {filteredEntities?.map((entity) => (
                  <tr 
                    key={entity.id} 
                    style={{ 
                      borderBottom: '1px solid #F0EDE5',
                      transition: 'background-color 0.2s',
                      cursor: 'pointer'
                    }}
                    className="hover:bg-gray-50"
                    onClick={() => setSelectedEntity(entity)}
                  >
                    <td style={{ padding: '12px' }}>
                      <span 
                        style={{ 
                          backgroundColor: getEntityTypeColor(entity.entity_type),
                          color: 'white',
                          padding: '2px 8px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '600',
                          textTransform: 'uppercase'
                        }}
                      >
                        {entity.entity_type}
                      </span>
                    </td>
                    <td style={{ padding: '12px', fontWeight: '500', color: '#2C3E50' }}>
                      {entity.entity_name}
                    </td>
                    <td style={{ 
                      padding: '12px', 
                      color: '#7F8C8D',
                      fontSize: '13px',
                      fontStyle: 'italic'
                    }}>
                      {entity.metadata?.context ? (
                        <div style={{ 
                          maxWidth: '400px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          "{entity.metadata.context}"
                        </div>
                      ) : (
                        <span style={{ opacity: 0.5 }}>No context available</span>
                      )}
                    </td>
                    <td style={{ 
                      padding: '12px',
                      textAlign: 'left',
                      color: '#7F8C8D'
                    }}>
                      {(entity.confidence * 100).toFixed(0)}%
                    </td>
                    <td style={{ 
                      padding: '12px',
                      textAlign: 'right'
                    }}>
                      <div style={{ display: 'flex', gap: '4px', justifyContent: 'flex-end' }}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowMetadataDialog(entity.id);
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
                            const newName = await prompt(
                              `Edit entity "${entity.entity_name}"\nType: ${entity.entity_type}`,
                              entity.entity_name
                            );
                            if (newName && newName !== entity.entity_name) {
                              updateEntityMutation.mutate({
                                entityId: entity.id,
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
                            const confirmed = await confirm(
                              `Delete entity "${entity.entity_name}"?\nType: ${entity.entity_type}`
                            );
                            if (confirmed) {
                              deleteEntityMutation.mutate(entity.id);
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
                ))}
              </tbody>
            </table>
            
            {filteredEntities?.length === 0 && (
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
                  {entities?.map(entity => (
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
                  {RELATIONSHIP_TYPES.map(type => (
                    <option key={type} value={type}>{type}</option>
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
                  {entities?.map(entity => (
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
                        confidence: 1.0,
                        manual: true,
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
                  const sourceEntity = getEntityById(rel.source_entity_id);
                  const targetEntity = getEntityById(rel.target_entity_id);
                  
                  return (
                    <tr key={rel.id} style={{ 
                      borderBottom: '1px solid #F0EDE5',
                      transition: 'background-color 0.2s'
                    }}
                    className="hover:bg-gray-50"
                    >
                      <td style={{ padding: '12px' }}>
                        <select
                          value={rel.source_entity_id}
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
                          {entities?.map(entity => (
                            <option key={entity.id} value={entity.id}>
                              {entity.entity_name} ({entity.entity_type})
                            </option>
                          ))}
                        </select>
                      </td>
                      <td style={{ padding: '12px' }}>
                        <select
                          value={rel.relationship_type}
                          onChange={(e) => updateRelationshipMutation.mutate({
                            id: rel.id,
                            relationship_type: e.target.value
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
                          {RELATIONSHIP_TYPES.map(type => (
                            <option key={type} value={type}>{type}</option>
                          ))}
                        </select>
                      </td>
                      <td style={{ padding: '12px' }}>
                        <select
                          value={rel.target_entity_id}
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
                          {entities?.map(entity => (
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
          entities={filteredEntities || []} 
          relationships={relationships || []} 
        />
      </div>

      {/* Metadata Dialog */}
      {showMetadataDialog && (() => {
        const entity = entities?.find(e => e.id === showMetadataDialog);
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
    </div>
  );
}