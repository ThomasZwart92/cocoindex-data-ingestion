'use client';

import { useState, useEffect, useRef } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { documentApi } from '@/lib/api';
import { useNotification } from '@/components/NotificationProvider';

interface Chunk {
  id: string;
  chunk_number: number;
  chunk_index?: number;
  chunk_text: string;
  chunk_size: number;
  start_position?: number;
  end_position?: number;
  parent_chunk_id?: string;
  chunk_level?: string;
  contextual_summary?: string;
  contextualized_text?: string;
  bm25_tokens?: string[];
  metadata?: any;
}

interface ChunkViewerProps {
  documentId: string;
  chunks: Chunk[];
  documentContent?: string;
}

export default function ChunkViewer({ documentId, chunks, documentContent }: ChunkViewerProps) {
  const queryClient = useQueryClient();
  const { notify, confirm } = useNotification();
  
  const [editingChunk, setEditingChunk] = useState<string | null>(null);
  const [chunkText, setChunkText] = useState<Record<string, string>>({});
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());
  const [showContext, setShowContext] = useState<Set<string>>(new Set());
  const [chunkingStrategy, setChunkingStrategy] = useState('recursive');
  const [chunkSize, setChunkSize] = useState(1500);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);

  // Initialize chunk text
  useState(() => {
    const textMap: Record<string, string> = {};
    chunks?.forEach(chunk => {
      textMap[chunk.id] = chunk.chunk_text;
    });
    setChunkText(textMap);
  });

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (openDropdown && !(event.target as HTMLElement).closest('.dropdown-container')) {
        setOpenDropdown(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [openDropdown]);

  // Group chunks by parent
  const chunkHierarchy = chunks?.reduce((acc, chunk) => {
    const parentId = chunk.parent_chunk_id || 'root';
    if (!acc[parentId]) acc[parentId] = [];
    acc[parentId].push(chunk);
    return acc;
  }, {} as Record<string, Chunk[]>);

  // Update chunk mutation
  const updateChunkMutation = useMutation({
    mutationFn: ({ chunkId, text }: { chunkId: string; text: string }) =>
      documentApi.updateChunk(chunkId, text),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chunks', documentId] });
      setEditingChunk(null);
      notify('Chunk updated successfully', 'success');
    },
    onError: (error) => {
      notify(`Failed to update chunk: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  // Delete chunk mutation
  const deleteChunkMutation = useMutation({
    mutationFn: documentApi.deleteChunk,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chunks', documentId] });
      notify('Chunk deleted successfully', 'success');
    },
    onError: (error) => {
      notify(`Failed to delete chunk: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  // Rechunk mutation
  const rechunkMutation = useMutation({
    mutationFn: () => 
      documentApi.rechunk(documentId, {
        strategy: chunkingStrategy,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['chunks', documentId] });
      notify('Document rechunked successfully', 'success');
    },
    onError: (error) => {
      notify(`Failed to rechunk: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  const toggleExpanded = (chunkId: string) => {
    const newExpanded = new Set(expandedChunks);
    if (newExpanded.has(chunkId)) {
      newExpanded.delete(chunkId);
    } else {
      newExpanded.add(chunkId);
    }
    setExpandedChunks(newExpanded);
  };

  const toggleContext = (chunkId: string) => {
    const newContext = new Set(showContext);
    if (newContext.has(chunkId)) {
      newContext.delete(chunkId);
    } else {
      newContext.add(chunkId);
    }
    setShowContext(newContext);
  };

  const getChunkContext = (chunk: Chunk) => {
    if (!documentContent || !chunk.start_position || !chunk.end_position) {
      return { before: '', after: '' };
    }
    
    const contextSize = 150;
    const before = documentContent.substring(
      Math.max(0, chunk.start_position - contextSize),
      chunk.start_position
    );
    const after = documentContent.substring(
      chunk.end_position,
      Math.min(documentContent.length, chunk.end_position + contextSize)
    );
    
    return { before, after };
  };

  const renderChunk = (chunk: Chunk, level: number = 0) => {
    const hasChildren = chunkHierarchy[chunk.id]?.length > 0;
    const isExpanded = expandedChunks.has(chunk.id);
    const hasContext = showContext.has(chunk.id);
    const context = getChunkContext(chunk);

    return (
      <div key={chunk.id} style={{
        border: '1px solid #E1E8ED',
        marginBottom: '12px',
        marginLeft: level > 0 ? '32px' : '0'
      }}>
        <div className="p-4">
          {/* Chunk header */}
          <div className="flex justify-between items-start mb-2">
            <div className="flex items-center space-x-2">
              {hasChildren && (
                <button
                  onClick={() => toggleExpanded(chunk.id)}
                  className="text-gray-600 hover:text-black"
                >
                  {isExpanded ? '▼' : '▶'}
                </button>
              )}
              <div>
                <span className="font-semibold" style={{ color: '#2C3E50' }}>
                  {chunk.chunk_level ? `${chunk.chunk_level.charAt(0).toUpperCase() + chunk.chunk_level.slice(1)} Chunk` : 
                   (level === 0 ? 'Parent' : 'Child') + ' Chunk'} #{chunk.chunk_index ?? chunk.chunk_number}
                </span>
                <span className="ml-4 text-gray-600">
                  {chunk.chunk_size || chunk.chunk_text?.length || 0} bytes
                  {hasChildren && ` | ${chunkHierarchy[chunk.id].length} children`}
                </span>
                {chunk.metadata?.confidence && (
                  <span className="ml-4 text-gray-600">
                    confidence: {(chunk.metadata.confidence * 100).toFixed(0)}%
                  </span>
                )}
                {chunk.bm25_tokens && (
                  <span className="ml-4 text-gray-600">
                    | BM25 tokens: {chunk.bm25_tokens.length}
                  </span>
                )}
              </div>
            </div>
            
            {/* Chunk actions */}
            <div className="relative">
              {editingChunk === chunk.id ? (
                <div className="space-x-2">
                  <button
                    onClick={() => {
                      updateChunkMutation.mutate({
                        chunkId: chunk.id,
                        text: chunkText[chunk.id]
                      });
                    }}
                    className="text-green-700 hover:text-green-900 hover:underline text-sm font-medium"
                  >
                    [Save]
                  </button>
                  <button
                    onClick={() => {
                      setEditingChunk(null);
                      setChunkText({ ...chunkText, [chunk.id]: chunk.chunk_text });
                    }}
                    className="text-gray-700 hover:text-gray-900 hover:underline text-sm font-medium"
                  >
                    [Cancel]
                  </button>
                </div>
              ) : (
                <div className="relative inline-block">
                  <button
                    onClick={() => setOpenDropdown(openDropdown === chunk.id ? null : chunk.id)}
                    className="text-blue-700 hover:text-blue-900 hover:underline text-sm font-medium flex items-center gap-1"
                  >
                    ⚙️ Options
                    <span className="text-xs">▼</span>
                  </button>
                  {openDropdown === chunk.id && (
                    <div className="absolute right-0 mt-1 w-48 bg-white border border-gray-300 rounded-md shadow-lg z-10">
                      <button
                        onClick={() => {
                          toggleContext(chunk.id);
                          setOpenDropdown(null);
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                      >
                        {hasContext ? '👁️ Hide Context' : '👁️ Show Context'}
                      </button>
                      <button
                        onClick={() => {
                          setEditingChunk(chunk.id);
                          setOpenDropdown(null);
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                      >
                        ✏️ Edit Chunk
                      </button>
                      <button
                        onClick={() => {
                          notify('Split feature coming soon', 'info');
                          setOpenDropdown(null);
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-400 hover:bg-gray-100"
                      >
                        ✂️ Split (Coming Soon)
                      </button>
                      <button
                        onClick={() => {
                          notify('Merge feature coming soon', 'info');
                          setOpenDropdown(null);
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-400 hover:bg-gray-100"
                      >
                        🔗 Merge (Coming Soon)
                      </button>
                      <hr className="my-1 border-gray-200" />
                      <button
                        onClick={async () => {
                          setOpenDropdown(null);
                          const confirmed = await confirm(
                            `Delete chunk #${chunk.chunk_number}?\nThis action cannot be undone.`
                          );
                          if (confirmed) {
                            deleteChunkMutation.mutate(chunk.id);
                          }
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                      >
                        🗑️ Delete Chunk
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Contextual Summary (from Anthropic's approach) */}
          {chunk.contextual_summary && (
            <div className="mb-3 p-3" style={{ 
              backgroundColor: '#E8F4FD',
              border: '1px solid #B8E0F7',
              borderRadius: '4px'
            }}>
              <div className="text-xs mb-1" style={{ color: '#2563EB', fontWeight: '600' }}>
                Contextual Summary (AI-Generated):
              </div>
              <div className="text-sm" style={{ color: '#1E40AF' }}>
                {chunk.contextual_summary}
              </div>
            </div>
          )}

          {/* Context display */}
          {hasContext && context.before && (
            <div className="mb-2 p-3" style={{ 
              backgroundColor: '#F8F7F3',
              border: '1px solid #E1E8ED'
            }}>
              <div className="text-xs mb-1" style={{ color: '#7F8C8D', fontWeight: '500' }}>Context Before:</div>
              <div className="text-sm opacity-50 font-mono">
                ...{context.before}
              </div>
            </div>
          )}

          {/* Chunk content */}
          {editingChunk === chunk.id ? (
            <textarea
              value={chunkText[chunk.id]}
              onChange={(e) => setChunkText({ ...chunkText, [chunk.id]: e.target.value })}
              className="w-full h-32 p-2 font-mono text-sm border border-gray-400"
            />
          ) : (
            <div className="whitespace-pre-wrap font-mono text-sm p-3" style={{ 
              border: '1px solid #F0F0F0',
              backgroundColor: '#FAFAFA',
              lineHeight: '1.6'
            }}>
              {chunk.chunk_text}
            </div>
          )}

          {/* Context after */}
          {hasContext && context.after && (
            <div className="mt-2 p-3" style={{ 
              backgroundColor: '#F8F7F3',
              border: '1px solid #E1E8ED'
            }}>
              <div className="text-xs mb-1" style={{ color: '#7F8C8D', fontWeight: '500' }}>Context After:</div>
              <div className="text-sm opacity-50 font-mono">
                {context.after}...
              </div>
            </div>
          )}
        </div>

        {/* Render children if expanded */}
        {hasChildren && isExpanded && (
          <div className="border-t border-gray-300">
            {chunkHierarchy[chunk.id].map(childChunk => 
              renderChunk(childChunk, level + 1)
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Chunk hierarchy display or document content */}
      {chunks && chunks.length > 0 ? (
        chunkHierarchy['root']?.map(chunk => renderChunk(chunk, 0)) || 
        chunks?.map(chunk => renderChunk(chunk, 0))
      ) : (
        <div style={{ 
          border: '1px solid #E1E8ED',
          padding: '20px'
        }}>
          <div className="mb-3">
            <span className="font-semibold" style={{ color: '#2C3E50' }}>
              Document Content (Not Chunked Yet)
            </span>
            <span className="ml-4 text-sm text-gray-600">
              Process this document to generate chunks
            </span>
          </div>
          {documentContent ? (
            <div className="whitespace-pre-wrap font-mono text-sm p-4" style={{ 
              border: '1px solid #F0F0F0',
              backgroundColor: '#FAFAFA',
              lineHeight: '1.6',
              maxHeight: '600px',
              overflowY: 'auto'
            }}>
              {documentContent}
            </div>
          ) : (
            <div className="text-gray-500 text-sm p-4" style={{
              backgroundColor: '#F8F7F3',
              border: '1px solid #E1E8ED'
            }}>
              No content available. The document may still be loading or processing.
            </div>
          )}
        </div>
      )}
    </div>
  );
}