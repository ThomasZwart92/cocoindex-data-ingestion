'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
});

interface Entity {
  id: string;
  entity_name: string;
  entity_type: string;
  confidence: number;
  metadata?: Record<string, any>;
}

interface Relationship {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relationship_type: string;
  confidence: number;
  relationship_label?: string;
}

interface GraphPreviewProps {
  entities: Entity[];
  relationships: Relationship[];
}

// Entity type colors matching the main component
const ENTITY_TYPE_COLORS: Record<string, string> = {
  'person': '#8B5CF6',
  'organization': '#6366F1',
  'component': '#EF4444',
  'product': '#F97316',
  'system': '#F59E0B',
  'technology': '#EAB308',
  'chemical': '#10B981',
  'measurement': '#14B8A6',
  'specification': '#06B6D4',
  'procedure': '#3B82F6',
  'concept': '#8B5CF6',
  'event': '#EC4899',
  'problem': '#DC2626',
  'state': '#7C3AED',
  'condition': '#A78BFA',
  'location': '#84CC16',
  'date': '#737373',
  'other': '#9CA3AF',
};

const getEntityColor = (type: string): string => {
  return ENTITY_TYPE_COLORS[type?.toLowerCase()] || ENTITY_TYPE_COLORS['other'];
};

export default function GraphPreview({ entities, relationships }: GraphPreviewProps) {
  console.log('GraphPreview component props:', { 
    entities: entities?.length || 0, 
    relationships: relationships?.length || 0,
    entitiesSample: entities?.slice(0, 2),
    relationshipsSample: relationships?.slice(0, 2)
  });
  
  const fgRef = useRef<any>();
  const searchContainerRef = useRef<HTMLDivElement>(null);
  const [graphData, setGraphData] = useState<any>({ nodes: [], links: [] });
  const [dimensions, setDimensions] = useState({ width: 800, height: 1200 });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [hoveredNodeData, setHoveredNodeData] = useState<Entity | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [focusedEntities, setFocusedEntities] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isUntangling, setIsUntangling] = useState(false);
  const [showTooltip, setShowTooltip] = useState(true);
  
  // Graph customization controls - stronger forces for better separation
  const [nodeSize, setNodeSize] = useState(0.8);
  const [linkDistance, setLinkDistance] = useState(150);  // Increased from 120
  const [chargeStrength, setChargeStrength] = useState(-800);  // Much stronger repulsion (was -240)

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
        setSearchTerm('');
      }
    };

    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  useEffect(() => {
    console.log('GraphPreview data:', { entities: entities?.length, relationships: relationships?.length });
    
    let filteredEntities = entities;
    let filteredRelationships = relationships;
    
    // If entities are focused, filter to show only those entities and their connections
    if (focusedEntities.size > 0) {
      // Find all relationships involving any of the focused entities
      filteredRelationships = relationships.filter(rel => 
        focusedEntities.has(rel.source_entity_id) || 
        focusedEntities.has(rel.target_entity_id)
      );
      
      // If there are relationships, show only connected entities
      if (filteredRelationships.length > 0) {
        // Get all entity IDs involved in these relationships
        const involvedEntityIds = new Set(focusedEntities);
        filteredRelationships.forEach(rel => {
          involvedEntityIds.add(rel.source_entity_id);
          involvedEntityIds.add(rel.target_entity_id);
        });
        
        // Filter entities to only those involved
        filteredEntities = entities.filter(entity => 
          involvedEntityIds.has(entity.id)
        );
      }
      // If no relationships, keep all entities visible but highlight the focused ones
      // This prevents the graph from becoming empty when focusing on an isolated node
    }
    
    // Convert entities to nodes
    const nodes = filteredEntities.map(entity => ({
      id: entity.id,
      name: entity.entity_name,
      type: entity.entity_type,
      confidence: entity.confidence,
      color: focusedEntities.has(entity.id) ? '#2563EB' : getEntityColor(entity.entity_type),
      val: (10 + (entity.confidence * 10)) * nodeSize * (focusedEntities.has(entity.id) ? 1.5 : 1), // Larger if focused
    }));

    // Create a set of valid entity IDs for quick lookup
    const validEntityIds = new Set(filteredEntities.map(e => e.id));

    // Convert relationships to links, filtering out invalid references
    const links = filteredRelationships
      .filter(rel => {
        const isValid = validEntityIds.has(rel.source_entity_id) && validEntityIds.has(rel.target_entity_id);
        if (!isValid) {
          console.warn(`Skipping invalid relationship: source=${rel.source_entity_id}, target=${rel.target_entity_id}`);
        }
        return isValid;
      })
      .map(rel => ({
        source: rel.source_entity_id,
        target: rel.target_entity_id,
        type: rel.relationship_label || rel.relationship_type,
        confidence: rel.confidence,
        color: `rgba(107, 114, 128, ${0.2 + rel.confidence * 0.6})`, // Opacity based on confidence
        width: 1 + rel.confidence * 2, // Width based on confidence
      }));

    console.log('GraphPreview processed:', { nodes: nodes.length, links: links.length });
    setGraphData({ nodes, links });
  }, [entities, relationships, nodeSize, focusedEntities]);
  
  // Update force graph physics when controls change
  useEffect(() => {
    if (fgRef.current) {
      // Update link distance
      if (fgRef.current.d3Force('link')) {
        fgRef.current.d3Force('link').distance(linkDistance);
      }
      
      // Update charge strength with distance-based falloff
      // Stronger repulsion for better node separation
      if (fgRef.current.d3Force('charge')) {
        fgRef.current.d3Force('charge')
          .strength(chargeStrength)
          .distanceMax(300); // Apply charge within wider radius for better separation
      }
      
      // Update collision force for better spacing
      if (fgRef.current.d3Force('collide')) {
        fgRef.current.d3Force('collide')
          .radius((node: any) => (node.val || 10) + 15) // More space between nodes
          .strength(1)
          .iterations(3); // More iterations for better collision detection
      }
      
      // Reheat the simulation to apply changes
      fgRef.current.d3ReheatSimulation();
    }
  }, [linkDistance, chargeStrength, nodeSize]); // Add nodeSize to dependencies

  useEffect(() => {
    // Update dimensions on mount and resize
    const updateDimensions = () => {
      const container = document.getElementById('graph-container');
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: 1200,  // Match the container height
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Force initial physics configuration after graph mounts
  useEffect(() => {
    if (fgRef.current && graphData.nodes.length > 0) {
      // Small delay to ensure graph is fully initialized
      setTimeout(() => {
        if (fgRef.current) {
          // Zoom to fit all nodes in view
          fgRef.current.zoomToFit(400, 50); // 400ms animation, 50px padding
          
          // Apply initial forces with stronger settings for better node separation
          if (fgRef.current.d3Force('link')) {
            fgRef.current.d3Force('link')
              .distance(linkDistance)
              .strength(0.5);  // Add link strength
          }
          if (fgRef.current.d3Force('charge')) {
            fgRef.current.d3Force('charge')
              .strength(chargeStrength)
              .distanceMax(300);  // Increased from 80 for longer range repulsion
          }
          if (fgRef.current.d3Force('collide')) {
            fgRef.current.d3Force('collide')
              .radius((node: any) => (node.val || 10) + 15)  // More space between nodes
              .strength(1)
              .iterations(3);  // More iterations for better collision detection
          }
          // Add center force to keep graph centered
          if (fgRef.current.d3Force('center')) {
            fgRef.current.d3Force('center')
              .x(dimensions.width / 2)
              .y(dimensions.height / 2);
          }
          // Reheat with more energy for better initial spread
          fgRef.current.d3ReheatSimulation();
          
          // Use the d3 force simulation directly to set alpha target
          const forceEngine = fgRef.current.d3Force;
          if (forceEngine) {
            // Keep simulation running longer for better spread
            fgRef.current.d3ReheatSimulation();
          }
        }
      }, 500);
    }
  }, [graphData]);

  // Add custom wheel handler for Ctrl+scroll zoom
  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey && fgRef.current) {
        e.preventDefault();
        const zoomSpeed = 0.002;
        const direction = e.deltaY < 0 ? 1 : -1;
        const factor = 1 + (direction * zoomSpeed * Math.abs(e.deltaY));
        fgRef.current.zoom(fgRef.current.zoom() * factor);
      }
    };

    const container = document.getElementById('graph-container');
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false });
    }

    return () => {
      if (container) {
        container.removeEventListener('wheel', handleWheel);
      }
    };
  }, []);

  const handleNodeClick = (node: any) => {
    setSelectedNode(node.id === selectedNode ? null : node.id);
    
    // Center the graph on the clicked node with 20% more zoom
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 1000);
      fgRef.current.zoom(2.4, 1000); // Increased from 2 to 2.4 (20% more)
    }
  };

  const handleBackgroundClick = () => {
    setSelectedNode(null);
    if (fgRef.current) {
      fgRef.current.zoomToFit(400);
    }
  };

  const resetNodePositions = () => {
    if (fgRef.current) {
      // Access the graph data directly through the graph's internal state
      const fg = fgRef.current;
      
      // Release all fixed positions by iterating through nodes
      graphData.nodes.forEach((node: any) => {
        // Clear fixed positions
        delete node.fx;
        delete node.fy;
      });
      
      // Update the graph data to trigger re-render
      setGraphData({ ...graphData });
      
      // Reheat the simulation to let nodes find natural positions
      setTimeout(() => {
        if (fgRef.current) {
          fgRef.current.d3ReheatSimulation();
        }
      }, 100);
    }
  };

  const untangleNodes = () => {
    if (!fgRef.current || isUntangling) return;
    
    setIsUntangling(true);
    
    // Clear all fixed positions first
    graphData.nodes.forEach((node: any) => {
      delete node.fx;
      delete node.fy;
    });
    
    // Apply strong repulsion forces temporarily
    if (fgRef.current.d3Force('charge')) {
      fgRef.current.d3Force('charge')
        .strength(-1000)  // Very strong repulsion
        .distanceMax(300); // Larger effect radius
    }
    
    // Increase collision detection
    if (fgRef.current.d3Force('collide')) {
      fgRef.current.d3Force('collide')
        .radius((node: any) => (node.val || 10) + 20) // Much larger collision radius
        .strength(1)
        .iterations(3); // More iterations for better collision resolution
    }
    
    // Add center force to prevent drift
    if (!fgRef.current.d3Force('center')) {
      // Create center force if it doesn't exist
      const d3 = fgRef.current.d3Force.constructor;
      fgRef.current.d3Force('center', d3.forceCenter(dimensions.width / 2, dimensions.height / 2));
    }
    
    // Increase link distance temporarily
    if (fgRef.current.d3Force('link')) {
      fgRef.current.d3Force('link')
        .distance(200) // Larger spacing
        .strength(0.5);
    }
    
    // Reheat simulation with high energy
    fgRef.current.d3ReheatSimulation();
    
    // After 2.5 seconds, gradually reduce forces back to normal
    setTimeout(() => {
      if (fgRef.current) {
        // Return to normal forces
        if (fgRef.current.d3Force('charge')) {
          fgRef.current.d3Force('charge')
            .strength(chargeStrength)
            .distanceMax(80);
        }
        
        if (fgRef.current.d3Force('collide')) {
          fgRef.current.d3Force('collide')
            .radius((node: any) => (node.val || 10) + 5)
            .strength(1)
            .iterations(1);
        }
        
        if (fgRef.current.d3Force('link')) {
          fgRef.current.d3Force('link')
            .distance(linkDistance)
            .strength(0.3);
        }
        
        // Remove center force
        if (fgRef.current.d3Force('center')) {
          fgRef.current.d3Force('center', null);
        }
        
        // Trigger another reheat to apply the normal forces
        fgRef.current.d3ReheatSimulation();
      }
      
      setIsUntangling(false);
    }, 2500);
  };

  const nodeCanvasObject = (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.name;
    const fontSize = 12 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    const textWidth = ctx.measureText(label).width;
    const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

    // Draw node circle
    ctx.fillStyle = node.color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
    ctx.fill();

    // Add border for selected/hovered nodes
    if (node.id === selectedNode || node.id === hoveredNode) {
      ctx.strokeStyle = node.id === selectedNode ? '#2563EB' : '#6B7280';
      ctx.lineWidth = node.id === selectedNode ? 3 : 2;
      ctx.stroke();
    }

    // Draw label
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#2C3E50';
    ctx.fillText(label, node.x, node.y + node.val + fontSize);
  };

  const linkCanvasObject = (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const start = link.source;
    const end = link.target;
    
    if (start && end && typeof start === 'object' && typeof end === 'object') {
      // Draw the link line (for both arrow and non-arrow modes)
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const angle = Math.atan2(dy, dx);
      
      // Calculate start and end positions accounting for node radius
      const startNodeRadius = start.val || 10;
      const endNodeRadius = end.val || 10;
      const lineStartX = start.x + Math.cos(angle) * startNodeRadius;
      const lineStartY = start.y + Math.sin(angle) * startNodeRadius;
      const lineEndX = end.x - Math.cos(angle) * endNodeRadius;
      const lineEndY = end.y - Math.sin(angle) * endNodeRadius;
      
      // Draw the connection line
      ctx.strokeStyle = link.color || 'rgba(107, 114, 128, 0.4)';
      ctx.lineWidth = link.width || 1;
      ctx.beginPath();
      ctx.moveTo(lineStartX, lineStartY);
      ctx.lineTo(lineEndX, lineEndY);
      ctx.stroke();
      
      // Draw arrow
      // Draw larger, more visible arrowhead
      const arrowLength = 10; // Fixed size for consistency
      const arrowAngle = Math.PI / 5; // Wider angle for better visibility
      
      // Position arrow at the end of the line
      const arrowTipX = lineEndX;
      const arrowTipY = lineEndY;
      
      ctx.fillStyle = link.color || 'rgba(107, 114, 128, 0.7)';
      ctx.strokeStyle = link.color || 'rgba(107, 114, 128, 0.7)';
      ctx.lineWidth = 1;
      
      // Draw filled triangle arrowhead
      ctx.beginPath();
      ctx.moveTo(arrowTipX, arrowTipY);
      ctx.lineTo(
        arrowTipX - arrowLength * Math.cos(angle - arrowAngle),
        arrowTipY - arrowLength * Math.sin(angle - arrowAngle)
      );
      ctx.lineTo(
        arrowTipX - arrowLength * Math.cos(angle + arrowAngle),
        arrowTipY - arrowLength * Math.sin(angle + arrowAngle)
      );
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      
      // Draw relationship type label
      const label = link.type;
      if (label && globalScale > 0.5) { // Only show labels when zoomed in enough
        const fontSize = 10 / globalScale;
        ctx.font = `${fontSize}px Sans-Serif`;
        
        const textPos = {
          x: start.x + (end.x - start.x) * 0.5,
          y: start.y + (end.y - start.y) * 0.5,
        };

        // Background for text
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fillRect(
          textPos.x - textWidth / 2 - 2,
          textPos.y - fontSize / 2 - 2,
          textWidth + 4,
          fontSize + 4
        );

        // Text
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#6B7280';
        ctx.fillText(label, textPos.x, textPos.y);
      }
    }
  };

  return (
    <div id="graph-container" style={{ 
      width: '100%', 
      height: '1200px',  // Full height as requested
      minHeight: '600px',  // Minimum height
      backgroundColor: '#F8F7F3',
      borderRadius: '8px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {graphData.nodes.length > 0 ? (
        <>
          {/* Debug info - remove after fixing */}
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            zIndex: 100,
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            padding: '8px',
            borderRadius: '4px',
            fontSize: '12px',
            border: '1px solid #E1E8ED'
          }}>
            Debug: {graphData.nodes.length} nodes, {graphData.links.length} links
            <br />
            Entities passed: {entities.length}, Relationships passed: {relationships.length}
          </div>
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height}
            nodeLabel="name"
            nodeColor="color"
            nodeVal="val"
            linkColor={() => 'transparent'}
            linkWidth={() => 0}
            linkDirectionalArrowLength={0}
            linkDirectionalArrowRelPos={1}
            linkCurvature={0.2}
            onNodeClick={handleNodeClick}
            onBackgroundClick={handleBackgroundClick}
            onNodeHover={(node: any, prevNode: any) => {
              setHoveredNode(node?.id || null);
              if (node) {
                const entity = entities.find(e => e.id === node.id);
                setHoveredNodeData(entity || null);
                // Get screen coordinates for tooltip
                const coords = fgRef.current?.graph2ScreenCoords(node.x, node.y);
                if (coords) {
                  setTooltipPosition({ x: coords.x, y: coords.y });
                }
              } else {
                setHoveredNodeData(null);
              }
            }}
            nodeCanvasObject={nodeCanvasObject}
            linkCanvasObject={linkCanvasObject}
            enableZoomInteraction={false}
            enablePanInteraction={true}
            enableNodeDrag={true}
            cooldownTicks={50}
            d3VelocityDecay={0.4}
            d3AlphaDecay={0.02}
            warmupTicks={100}
            d3Force={{
              link: { distance: linkDistance, strength: 0.3 },
              charge: { strength: chargeStrength, distanceMax: 80 },
              collide: { radius: (node: any) => (node.val || 10) + 5, strength: 1 },
              x: { strength: 0 },
              y: { strength: 0 }
            }}
            onNodeDragStart={(node: any) => {
              // Node drag start - no alpha target changes needed
              node.__initialDrag = true;
            }}
            onNodeDrag={(node: any) => {
              // Keep node fixed to cursor position during drag
              node.fx = node.x;
              node.fy = node.y;
            }}
            onNodeDragEnd={(node: any) => {
              // Pin the node at its current position after dragging
              node.fx = node.x;
              node.fy = node.y;
              node.__initialDrag = false;
            }}
          />
          
          {/* Entity Focus Selector */}
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: 'white',
            padding: '10px',
            borderRadius: '6px',
            border: '1px solid #E1E8ED',
            fontSize: '13px',
            zIndex: 10,
            width: '700px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ color: '#6B7280', fontWeight: '500', flexShrink: 0 }}>Focus on:</label>
              
              {/* Multi-select with search */}
              <div ref={searchContainerRef} style={{ position: 'relative', width: '300px' }}>
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  onFocus={() => setIsDropdownOpen(true)}
                  placeholder={focusedEntities.size > 0 ? `${focusedEntities.size} selected` : "Search entities..."}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    border: '1px solid #E1E8ED',
                    borderRadius: '4px',
                    backgroundColor: 'white',
                    color: '#2C3E50',
                    fontSize: '13px'
                  }}
                />
                
                {isDropdownOpen && (
                  <div style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    right: 0,
                    marginTop: '4px',
                    backgroundColor: 'white',
                    border: '1px solid #E1E8ED',
                    borderRadius: '4px',
                    maxHeight: '300px',
                    overflowY: 'auto',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                  }}>
                    {entities
                      .filter(entity => 
                        entity.entity_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                        entity.entity_type.toLowerCase().includes(searchTerm.toLowerCase())
                      )
                      .sort((a, b) => a.entity_name.localeCompare(b.entity_name))
                      .map(entity => (
                        <div
                          key={entity.id}
                          onClick={() => {
                            const newFocused = new Set(focusedEntities);
                            if (newFocused.has(entity.id)) {
                              newFocused.delete(entity.id);
                            } else {
                              newFocused.add(entity.id);
                            }
                            setFocusedEntities(newFocused);
                          }}
                          style={{
                            padding: '8px 12px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            backgroundColor: focusedEntities.has(entity.id) ? '#EEF2FF' : 'transparent',
                            borderLeft: focusedEntities.has(entity.id) ? '3px solid #6366F1' : '3px solid transparent'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.backgroundColor = focusedEntities.has(entity.id) ? '#E0E7FF' : '#F9FAFB';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.backgroundColor = focusedEntities.has(entity.id) ? '#EEF2FF' : 'transparent';
                          }}
                        >
                          <div>
                            <div style={{ color: '#2C3E50', fontWeight: focusedEntities.has(entity.id) ? '600' : '400' }}>
                              {entity.entity_name}
                            </div>
                            <div style={{ color: '#6B7280', fontSize: '11px' }}>
                              {entity.entity_type}
                            </div>
                          </div>
                          {focusedEntities.has(entity.id) && (
                            <span style={{ color: '#6366F1' }}>✓</span>
                          )}
                        </div>
                      ))}
                    {entities.filter(e => 
                      e.entity_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                      e.entity_type.toLowerCase().includes(searchTerm.toLowerCase())
                    ).length === 0 && (
                      <div style={{ padding: '12px', color: '#6B7280', textAlign: 'center' }}>
                        No entities found
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              {/* Close dropdown button */}
              {isDropdownOpen && (
                <button
                  onClick={() => {
                    setIsDropdownOpen(false);
                    setSearchTerm('');
                  }}
                  style={{
                    padding: '6px',
                    backgroundColor: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#6B7280'
                  }}
                >
                  ✕
                </button>
              )}
              
              <span style={{ 
                color: '#6B7280', 
                fontSize: '12px',
                padding: '4px 8px',
                backgroundColor: '#F9FAFB',
                borderRadius: '4px',
                minWidth: '140px',
                textAlign: 'center',
                flexShrink: 0
              }}>
                {focusedEntities.size > 0 
                  ? `Showing ${graphData.nodes.length} of ${entities.length} entities`
                  : `${entities.length} entities total`
                }
              </span>
              <button
                onClick={() => {
                  setFocusedEntities(new Set());
                  setSearchTerm('');
                }}
                disabled={focusedEntities.size === 0}
                style={{
                  padding: '6px 12px',
                  backgroundColor: focusedEntities.size > 0 ? '#F3F4F6' : '#F9FAFB',
                  border: '1px solid #E1E8ED',
                  borderRadius: '4px',
                  cursor: focusedEntities.size > 0 ? 'pointer' : 'default',
                  color: focusedEntities.size > 0 ? '#6B7280' : '#D1D5DB',
                  fontSize: '12px',
                  flexShrink: 0
                }}
                title="Show all entities"
              >
                Clear All
              </button>
            </div>
            
            {/* Selected entities chips */}
            {focusedEntities.size > 0 && (
              <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                {Array.from(focusedEntities).map(entityId => {
                  const entity = entities.find(e => e.id === entityId);
                  if (!entity) return null;
                  return (
                    <div
                      key={entityId}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: '#EEF2FF',
                        border: '1px solid #C7D2FE',
                        borderRadius: '4px',
                        fontSize: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                    >
                      <span style={{ color: '#4F46E5' }}>{entity.entity_name}</span>
                      <button
                        onClick={() => {
                          const newFocused = new Set(focusedEntities);
                          newFocused.delete(entityId);
                          setFocusedEntities(newFocused);
                        }}
                        style={{
                          backgroundColor: 'transparent',
                          border: 'none',
                          color: '#6B7280',
                          cursor: 'pointer',
                          padding: '0',
                          fontSize: '14px'
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          
          {/* Controls Panel */}
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            backgroundColor: 'white',
            padding: '12px',
            borderRadius: '6px',
            border: '1px solid #E1E8ED',
            fontSize: '12px',
            minWidth: '250px'
          }}>
            <div style={{ fontWeight: '600', marginBottom: '8px', color: '#2C3E50' }}>
              Graph Controls
            </div>
            
            {/* Node Size Control */}
            <div style={{ marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ color: '#6B7280' }}>Node Size</span>
                <span style={{ color: '#2C3E50', fontWeight: '500' }}>{nodeSize.toFixed(1)}x</span>
              </label>
              <input
                type="range"
                min="0.5"
                max="3"
                step="0.1"
                value={nodeSize}
                onChange={(e) => setNodeSize(parseFloat(e.target.value))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
            </div>
            
            {/* Link Distance Control */}
            <div style={{ marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ color: '#6B7280' }}>Node Spacing</span>
                <span style={{ color: '#2C3E50', fontWeight: '500' }}>{linkDistance}</span>
              </label>
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={linkDistance}
                onChange={(e) => setLinkDistance(parseInt(e.target.value))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
            </div>
            
            {/* Charge Strength Control */}
            <div style={{ marginBottom: '8px' }}>
              <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ color: '#6B7280' }}>Repulsion Force</span>
                <span style={{ color: '#2C3E50', fontWeight: '500' }}>{Math.abs(chargeStrength)}</span>
              </label>
              <input
                type="range"
                min="-500"
                max="-10"
                step="10"
                value={chargeStrength}
                onChange={(e) => setChargeStrength(parseInt(e.target.value))}
                style={{ width: '100%', cursor: 'pointer' }}
              />
            </div>
            
            {/* Show Tooltip Toggle */}
            <div style={{ 
              marginBottom: '8px',
              paddingTop: '8px',
              borderTop: '1px solid #E1E8ED'
            }}>
              <label style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                cursor: 'pointer'
              }}>
                <span style={{ color: '#6B7280' }}>Show Tooltip</span>
                <input
                  type="checkbox"
                  checked={showTooltip}
                  onChange={(e) => setShowTooltip(e.target.checked)}
                  style={{ cursor: 'pointer' }}
                />
              </label>
            </div>
            
            {/* Zoom Controls */}
            <div style={{ borderTop: '1px solid #E1E8ED', paddingTop: '8px', marginTop: '8px' }}>
              <div style={{ fontSize: '10px', color: '#9CA3AF', marginBottom: '6px', textAlign: 'center' }}>
                Tip: Use Ctrl+Scroll to zoom
              </div>
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                onClick={() => fgRef.current?.zoomToFit(400)}
                style={{
                  flex: 1,
                  padding: '4px 8px',
                  backgroundColor: '#F8F7F3',
                  border: '1px solid #E1E8ED',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  color: '#2C3E50'
                }}
                title="Fit to view"
              >
                Fit
              </button>
              <button
                onClick={() => fgRef.current?.zoom(1.5, 200)}
                style={{
                  flex: 1,
                  padding: '4px 8px',
                  backgroundColor: '#F8F7F3',
                  border: '1px solid #E1E8ED',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  color: '#2C3E50'
                }}
                title="Zoom in"
              >
                Zoom In
              </button>
              <button
                onClick={() => fgRef.current?.zoom(0.67, 200)}
                style={{
                  flex: 1,
                  padding: '4px 8px',
                  backgroundColor: '#F8F7F3',
                  border: '1px solid #E1E8ED',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  color: '#2C3E50'
                }}
                title="Zoom out"
              >
                Zoom Out
              </button>
              </div>
            </div>
            
            {/* Action Buttons */}
            <div style={{ marginTop: '8px', display: 'flex', gap: '4px' }}>
              <button
                onClick={untangleNodes}
                disabled={isUntangling}
                style={{
                  flex: 1,
                  padding: '6px 8px',
                  backgroundColor: isUntangling ? '#9CA3AF' : '#10B981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isUntangling ? 'wait' : 'pointer',
                  fontWeight: '500'
                }}
                title="Automatically untangle overlapping nodes"
              >
                {isUntangling ? 'Untangling...' : 'Untangle'}
              </button>
              <button
                onClick={resetNodePositions}
                style={{
                  flex: 1,
                  padding: '6px 8px',
                  backgroundColor: '#6366F1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontWeight: '500'
                }}
                title="Reset all node positions"
              >
                Reset Positions
              </button>
            </div>
          </div>

          {/* Legend */}
          <div style={{
            position: 'absolute',
            bottom: '10px',
            left: '10px',
            backgroundColor: 'white',
            padding: '8px',
            borderRadius: '6px',
            border: '1px solid #E1E8ED',
            fontSize: '11px',
            minWidth: '150px',
            maxWidth: '250px',
            maxHeight: 'calc(100% - 20px)', // Full height minus margins
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ 
              fontWeight: '600', 
              marginBottom: '4px', 
              color: '#2C3E50',
              flexShrink: 0 
            }}>
              Entity Types ({Array.from(new Set(entities.map(e => e.entity_type))).length}):
            </div>
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column',
              gap: '4px',
              overflowY: 'auto',
              overflowX: 'hidden',
              flex: 1
            }}>
              {Array.from(new Set(entities.map(e => e.entity_type)))
                .sort((a, b) => a.localeCompare(b))
                .map(type => (
                <div key={type} style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '6px',
                  padding: '2px 0',
                  flexShrink: 0
                }}>
                  <div style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    backgroundColor: getEntityColor(type),
                    flexShrink: 0
                  }} />
                  <span style={{ 
                    color: '#6B7280', 
                    textTransform: 'capitalize',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}>
                    {type}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Selected node info */}
          {selectedNode && (
            <div style={{
              position: 'absolute',
              top: focusedEntities.size > 0 ? '120px' : '10px',
              left: '10px',
              backgroundColor: 'white',
              padding: '12px',
              borderRadius: '6px',
              border: '1px solid #E1E8ED',
              maxWidth: '250px'
            }}>
              <div style={{ fontWeight: '600', marginBottom: '4px', color: '#2C3E50' }}>
                {entities.find(e => e.id === selectedNode)?.entity_name}
              </div>
              <div style={{ fontSize: '12px', color: '#6B7280' }}>
                Type: {entities.find(e => e.id === selectedNode)?.entity_type}
              </div>
              <div style={{ fontSize: '12px', color: '#6B7280' }}>
                Confidence: {((entities.find(e => e.id === selectedNode)?.confidence || 0) * 100).toFixed(0)}%
              </div>
              <div style={{ fontSize: '12px', color: '#6B7280', marginTop: '4px' }}>
                Relationships: {relationships.filter(r => 
                  r.source_entity_id === selectedNode || r.target_entity_id === selectedNode
                ).length}
              </div>
              {!focusedEntities.has(selectedNode) && (
                <button
                  onClick={() => {
                    const newFocused = new Set(focusedEntities);
                    newFocused.add(selectedNode);
                    setFocusedEntities(newFocused);
                  }}
                  style={{
                    marginTop: '8px',
                    width: '100%',
                    padding: '4px 8px',
                    backgroundColor: '#6366F1',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  Focus on this entity
                </button>
              )}
            </div>
          )}

          {/* Hover Tooltip */}
          {showTooltip && hoveredNodeData && hoveredNode !== selectedNode && (
            <div style={{
              position: 'absolute',
              left: tooltipPosition.x + 10,
              top: tooltipPosition.y - 10,
              backgroundColor: 'rgba(0, 0, 0, 0.9)',
              color: 'white',
              padding: '8px 12px',
              borderRadius: '6px',
              fontSize: '12px',
              zIndex: 1000,
              pointerEvents: 'none',
              maxWidth: '300px',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
              transform: 'translate(-50%, -100%)'
            }}>
              <div style={{ fontWeight: '600', marginBottom: '4px' }}>
                {hoveredNodeData.entity_name}
              </div>
              <div style={{ fontSize: '11px', opacity: 0.9, marginBottom: '2px' }}>
                Type: {hoveredNodeData.entity_type}
              </div>
              <div style={{ fontSize: '11px', opacity: 0.9, marginBottom: '2px' }}>
                Confidence: {(hoveredNodeData.confidence * 100).toFixed(0)}%
              </div>
              
              {/* Show metadata if available */}
              {hoveredNodeData.metadata && (
                <>
                  {hoveredNodeData.metadata.context && (
                    <div style={{ 
                      fontSize: '11px', 
                      opacity: 0.9, 
                      marginTop: '6px',
                      borderTop: '1px solid rgba(255, 255, 255, 0.2)',
                      paddingTop: '6px',
                      fontStyle: 'italic'
                    }}>
                      "{hoveredNodeData.metadata.context}"
                    </div>
                  )}
                  {hoveredNodeData.metadata.category && (
                    <div style={{ fontSize: '11px', opacity: 0.9, marginTop: '4px' }}>
                      Category: {hoveredNodeData.metadata.category}
                    </div>
                  )}
                  {hoveredNodeData.metadata.attributes && Object.keys(hoveredNodeData.metadata.attributes).length > 0 && (
                    <div style={{ fontSize: '11px', opacity: 0.9, marginTop: '4px' }}>
                      {Object.entries(hoveredNodeData.metadata.attributes).slice(0, 2).map(([key, value]) => (
                        <div key={key}>• {key}: {String(value)}</div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </>
      ) : (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#7F8C8D'
        }}>
          No entities or relationships to display
        </div>
      )}
    </div>
  );
}
