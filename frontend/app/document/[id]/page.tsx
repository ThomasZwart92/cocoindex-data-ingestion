'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentApi } from '@/lib/api';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useNotification } from '@/components/NotificationProvider';
import ChunkViewer from '@/components/document/ChunkViewer';
import EntitiesRelationships from '@/components/document/EntitiesRelationships';

type TabType = 'preview' | 'entities' | 'metadata' | 'status';

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const documentId = params.id as string;
  const { notify, confirm } = useNotification();

  const [activeTab, setActiveTab] = useState<TabType>('preview');
  const [metadata, setMetadata] = useState<Record<string, any>>({});
  const [isProcessing, setIsProcessing] = useState(false);

  // Fetch document
  const { data: document, isLoading: docLoading } = useQuery({
    queryKey: ['document', documentId],
    queryFn: async () => {
      const response = await documentApi.get(documentId);
      setMetadata(response.data.metadata || {});
      return response.data;
    },
  });

  // Fetch chunks
  const { data: chunks, isLoading: chunksLoading } = useQuery({
    queryKey: ['chunks', documentId],
    queryFn: async () => {
      const response = await documentApi.getChunks(documentId);
      return response.data;
    },
  });

  // Fetch entities
  const { data: entities } = useQuery({
    queryKey: ['entities', documentId],
    queryFn: async () => {
      const response = await documentApi.getEntities(documentId);
      return response.data;
    },
  });

  // Update document metadata
  const updateMetadataMutation = useMutation({
    mutationFn: () => documentApi.update(documentId, { metadata }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', documentId] });
      notify('Metadata saved successfully', 'success');
    },
    onError: (error: any) => {
      notify(`Failed to save metadata: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  // Process document mutation
  const processDocumentMutation = useMutation({
    mutationFn: () => {
      setIsProcessing(true);
      return documentApi.reprocess(documentId);
    },
    onSuccess: () => {
      notify('Document processing started', 'success');
      // Poll for status updates
      const pollInterval = setInterval(async () => {
        const response = await documentApi.get(documentId);
        if (response.data.status !== 'processing') {
          setIsProcessing(false);
          clearInterval(pollInterval);
          queryClient.invalidateQueries({ queryKey: ['document', documentId] });
          queryClient.invalidateQueries({ queryKey: ['chunks', documentId] });
          queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
          if (response.data.status === 'failed') {
            notify('Document processing failed', 'error');
          } else if (response.data.chunk_count > 0) {
            notify(`Processing complete: ${response.data.chunk_count} chunks created`, 'success');
          }
        }
      }, 2000); // Poll every 2 seconds
      
      // Stop polling after 60 seconds
      setTimeout(() => {
        clearInterval(pollInterval);
        setIsProcessing(false);
      }, 60000);
    },
    onError: (error: any) => {
      setIsProcessing(false);
      notify(`Failed to process document: ${error.message || 'Unknown error'}`, 'error');
    },
  });

  if (docLoading || chunksLoading) {
    return <div className="p-8">Loading document...</div>;
  }

  if (!document) {
    return <div className="p-8">Document not found</div>;
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <div style={{ borderBottom: '1px solid #E1E8ED', paddingTop: '24px', paddingBottom: '24px' }}>
        <div style={{ maxWidth: '85%', margin: '0 auto' }}>
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-xl font-bold" style={{ color: '#2C3E50' }}>
                Document: {document.title}
              </h1>
              <div className="text-sm text-gray-600 mt-2">
                Source: {document.source_type} | Status: {document.status} | 
                Security: {document.security_level} (Level {document.access_level})
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={async () => {
                  if (!isProcessing && document.status !== 'processing') {
                    const confirmed = await confirm(
                      'Process this document?\nThis will extract chunks, entities, and relationships.'
                    );
                    if (confirmed) {
                      processDocumentMutation.mutate();
                    }
                  }
                }}
                disabled={isProcessing || document.status === 'processing'}
                style={{
                  backgroundColor: isProcessing || document.status === 'processing' ? '#E8F4FD' : 'white',
                  color: isProcessing || document.status === 'processing' ? '#95A5A6' : '#3498DB',
                  border: '1px solid #BDD7ED',
                  padding: '8px 16px',
                  fontSize: '14px',
                  fontWeight: '500',
                  marginRight: '8px',
                  cursor: isProcessing || document.status === 'processing' ? 'not-allowed' : 'pointer',
                  opacity: isProcessing || document.status === 'processing' ? 0.7 : 1
                }}
                className={isProcessing || document.status === 'processing' ? '' : 'hover:bg-blue-50'}
              >
                {isProcessing || document.status === 'processing' ? (
                  <>
                    <span className="inline-block animate-spin mr-2">⏳</span>
                    Processing...
                  </>
                ) : document.chunk_count > 0 ? (
                  '🔄 Reprocess'
                ) : (
                  '📊 Process'
                )}
              </button>
              <button
                onClick={async () => {
                  const confirmed = await confirm(
                    'Save all changes?\nThis will save metadata, chunks, and entity modifications.'
                  );
                  if (confirmed) {
                    updateMetadataMutation.mutate();
                  }
                }}
                style={{
                  backgroundColor: '#3498DB',
                  color: 'white',
                  padding: '8px 16px',
                  fontSize: '14px',
                  fontWeight: '500'
                }}
                className="hover:opacity-90"
              >
                💾 Save All
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{ borderBottom: '1px solid #E1E8ED' }}>
        <div style={{ maxWidth: '85%', margin: '0 auto' }}>
          <div className="flex">
            <button
              onClick={() => setActiveTab('preview')}
              style={{
                padding: '12px 24px',
                borderRight: '1px solid #E1E8ED',
                backgroundColor: activeTab === 'preview' ? '#3498DB' : 'transparent',
                color: activeTab === 'preview' ? 'white' : '#2C3E50',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s'
              }}
              className="hover:bg-gray-50"
            >
              Preview
            </button>
            <button
              onClick={() => setActiveTab('entities')}
              style={{
                padding: '12px 24px',
                borderRight: '1px solid #E1E8ED',
                backgroundColor: activeTab === 'entities' ? '#3498DB' : 'transparent',
                color: activeTab === 'entities' ? 'white' : '#2C3E50',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s'
              }}
              className="hover:bg-gray-50"
            >
              Entities ({entities?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('metadata')}
              style={{
                padding: '12px 24px',
                borderRight: '1px solid #E1E8ED',
                backgroundColor: activeTab === 'metadata' ? '#3498DB' : 'transparent',
                color: activeTab === 'metadata' ? 'white' : '#2C3E50',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s'
              }}
              className="hover:bg-gray-50"
            >
              Metadata
            </button>
            <button
              onClick={() => setActiveTab('status')}
              style={{
                padding: '12px 24px',
                backgroundColor: activeTab === 'status' ? '#3498DB' : 'transparent',
                color: activeTab === 'status' ? 'white' : '#2C3E50',
                fontSize: '14px',
                fontWeight: '500',
                transition: 'all 0.2s'
              }}
              className="hover:bg-gray-50"
            >
              Status
            </button>
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div style={{ maxWidth: '85%', margin: '0 auto', paddingTop: '24px', paddingBottom: '24px' }}>
        {activeTab === 'preview' && (
          <>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold" style={{ color: '#2C3E50' }}>
                Document Chunks ({chunks?.length || 0})
              </h2>
              {(isProcessing || document.status === 'processing') && (
                <div className="flex items-center text-sm" style={{ color: '#3498DB' }}>
                  <span className="inline-block animate-spin mr-2">⏳</span>
                  Processing document...
                </div>
              )}
              {!isProcessing && document.status === 'failed' && chunks?.length === 0 && (
                <div className="text-sm" style={{ color: '#E74C3C' }}>
                  ⚠️ Processing failed - please try again
                </div>
              )}
              {!isProcessing && document.status === 'ingested' && chunks?.length > 0 && (
                <div className="text-sm" style={{ color: '#27AE60' }}>
                  ✅ Successfully processed
                </div>
              )}
            </div>
            <ChunkViewer
              documentId={documentId}
              chunks={chunks || []}
              documentContent={document.content}
            />
          </>
        )}

        {activeTab === 'entities' && (
          <EntitiesRelationships
            documentId={documentId}
            entities={entities || []}
          />
        )}

        {activeTab === 'metadata' && (
          <div className="space-y-6">
            <h2 className="text-lg font-bold mb-4" style={{ color: '#2C3E50' }}>Document Metadata</h2>
            
            {/* Core Metadata */}
            <div style={{ 
              border: '1px solid #E1E8ED',
              padding: '20px',
              marginBottom: '20px'
            }}>
              <h3 className="font-bold mb-4" style={{ color: '#2C3E50' }}>Core Information</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm mb-1">Document ID:</label>
                  <input
                    type="text"
                    value={document.id}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Source Type:</label>
                  <input
                    type="text"
                    value={document.source_type}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Source ID:</label>
                  <input
                    type="text"
                    value={document.source_id}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Mime Type:</label>
                  <input
                    type="text"
                    value={document.mime_type || metadata.mime_type || 'Not detected'}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Status:</label>
                  <input
                    type="text"
                    value={document.status}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Author:</label>
                  <input
                    type="text"
                    value={document.author || metadata.author || 'Unknown'}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Created:</label>
                  <input
                    type="text"
                    value={new Date(document.created_at).toLocaleString()}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Updated:</label>
                  <input
                    type="text"
                    value={new Date(document.updated_at).toLocaleString()}
                    disabled
                    className="w-full opacity-50"
                  />
                </div>
              </div>
            </div>

            {/* Editable Metadata */}
            <div style={{ 
              border: '1px solid #E1E8ED',
              padding: '20px',
              marginBottom: '20px'
            }}>
              <h3 className="font-bold mb-4" style={{ color: '#2C3E50' }}>Editable Metadata</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm mb-1">Author:</label>
                  <input
                    type="text"
                    value={metadata.author || ''}
                    onChange={(e) => setMetadata({ ...metadata, author: e.target.value })}
                    className="w-full"
                    placeholder="Enter author..."
                  />
                </div>
                <div>
                  <label className="block text-sm mb-1">Category:</label>
                  <select
                    value={metadata.category || ''}
                    onChange={(e) => setMetadata({ ...metadata, category: e.target.value })}
                    className="w-full"
                  >
                    <option value="">Select category...</option>
                    <optgroup label="Technical Documentation">
                      <option value="product_manual">Product Manual</option>
                      <option value="troubleshooting_guide">Troubleshooting Guide</option>
                      <option value="technical_specification">Technical Specification</option>
                      <option value="installation_guide">Installation Guide</option>
                      <option value="service_manual">Service Manual</option>
                    </optgroup>
                    <optgroup label="Business Documentation">
                      <option value="sop">Standard Operating Procedure</option>
                      <option value="policy">Policy Document</option>
                      <option value="training_material">Training Material</option>
                      <option value="meeting_notes">Meeting Notes</option>
                      <option value="report">Report</option>
                    </optgroup>
                    <optgroup label="Customer-Facing">
                      <option value="faq">FAQ</option>
                      <option value="user_guide">User Guide</option>
                      <option value="release_notes">Release Notes</option>
                      <option value="warranty_terms">Warranty Terms</option>
                      <option value="datasheet">Datasheet</option>
                    </optgroup>
                    <optgroup label="Internal Operations">
                      <option value="incident_report">Incident Report</option>
                      <option value="project_plan">Project Plan</option>
                      <option value="requirements">Requirements Document</option>
                      <option value="design_document">Design Document</option>
                      <option value="test_plan">Test Plan</option>
                    </optgroup>
                  </select>
                </div>
                <div>
                  <label className="block text-sm mb-1">Department:</label>
                  <select
                    value={metadata.department || ''}
                    onChange={(e) => setMetadata({ ...metadata, department: e.target.value })}
                    className="w-full"
                  >
                    <option value="">Select department...</option>
                    <option value="engineering">Engineering</option>
                    <option value="technical_support">Technical Support</option>
                    <option value="client_success">Client Success</option>
                    <option value="supply_chain">Supply Chain</option>
                    <option value="logistics">Logistics</option>
                    <option value="sales">Sales</option>
                    <option value="finance">Finance</option>
                    <option value="marketing">Marketing</option>
                    <option value="people_culture">People & Culture</option>
                    <option value="special_projects">Special Projects</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm mb-1">Version:</label>
                  <input
                    type="text"
                    value={metadata.version || ''}
                    onChange={(e) => setMetadata({ ...metadata, version: e.target.value })}
                    className="w-full"
                    placeholder="e.g., 1.0, 2.1..."
                  />
                </div>
                <div className="col-span-2">
                  <label className="block text-sm mb-1">Tags:</label>
                  <input
                    type="text"
                    value={metadata.tags || ''}
                    onChange={(e) => setMetadata({ ...metadata, tags: e.target.value })}
                    className="w-full"
                    placeholder="comma, separated, tags"
                  />
                </div>
                <div className="col-span-2">
                  <label className="block text-sm mb-1">Description:</label>
                  <textarea
                    value={metadata.description || ''}
                    onChange={(e) => setMetadata({ ...metadata, description: e.target.value })}
                    className="w-full h-24"
                    placeholder="Enter document description..."
                  />
                </div>
              </div>
            </div>

            {/* AI Metadata Extraction */}
            <div style={{ 
              border: '1px solid #BDD7ED',
              padding: '20px',
              backgroundColor: '#F8FBFF'
            }}>
              <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold" style={{ color: '#3498DB' }}>AI Metadata Extraction</h3>
                <button
                  onClick={async () => {
                    try {
                      notify('Extracting metadata with AI...', 'info');
                      const extractResponse = await documentApi.extractMetadata(documentId);
                      
                      // Wait a bit for extraction to complete
                      setTimeout(async () => {
                        try {
                          const suggestionsResponse = await documentApi.getSuggestedMetadata(documentId);
                          if (suggestionsResponse.data?.suggestions) {
                            setMetadata({
                              ...metadata,
                              ai_suggested: suggestionsResponse.data.suggestions,
                              confidence_scores: suggestionsResponse.data.confidence_scores
                            });
                            notify('Metadata extracted successfully!', 'success');
                          } else {
                            // Extraction might still be processing
                            notify('Metadata extraction in progress, please wait...', 'info');
                          }
                        } catch (suggestionError) {
                          console.error('Error fetching suggestions:', suggestionError);
                          // Don't show error here as extraction might still be processing
                        }
                      }, 3000);
                    } catch (error: any) {
                      console.error('Error starting extraction:', error);
                      notify(error.response?.data?.detail || 'Failed to extract metadata', 'error');
                    }
                  }}
                  style={{
                    backgroundColor: '#3498DB',
                    color: 'white',
                    padding: '6px 12px',
                    fontSize: '13px',
                    fontWeight: '500'
                  }}
                  className="hover:opacity-90"
                >
                  🤖 Extract Metadata
                </button>
              </div>
              
              {metadata.ai_suggested && (
                <div className="space-y-3">
                  {metadata.ai_suggested.category && (
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold">Category:</span>
                      <span className="text-sm">{metadata.ai_suggested.category}</span>
                      <button
                        onClick={() => {
                          setMetadata({ ...metadata, category: metadata.ai_suggested.category });
                          notify('Applied category suggestion', 'success');
                        }}
                        className="text-blue-600 hover:underline text-sm"
                      >
                        Apply
                      </button>
                    </div>
                  )}
                  
                  {metadata.ai_suggested.department && (
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold">Department:</span>
                      <span className="text-sm">{metadata.ai_suggested.department}</span>
                      <button
                        onClick={() => {
                          setMetadata({ ...metadata, department: metadata.ai_suggested.department });
                          notify('Applied department suggestion', 'success');
                        }}
                        className="text-blue-600 hover:underline text-sm"
                      >
                        Apply
                      </button>
                    </div>
                  )}
                  
                  {metadata.ai_suggested.tags && metadata.ai_suggested.tags.length > 0 && (
                    <div>
                      <span className="text-sm font-semibold">Tags:</span>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {metadata.ai_suggested.tags.map((tag: string) => (
                          <span
                            key={tag}
                            className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <button
                        onClick={() => {
                          setMetadata({ 
                            ...metadata, 
                            tags: metadata.ai_suggested.tags.join(', ') 
                          });
                          notify('Applied tag suggestions', 'success');
                        }}
                        className="text-blue-600 hover:underline text-sm mt-2"
                      >
                        Apply All Tags
                      </button>
                    </div>
                  )}
                  
                  {metadata.ai_suggested.description && (
                    <div>
                      <span className="text-sm font-semibold">Description:</span>
                      <p className="text-sm mt-1">{metadata.ai_suggested.description}</p>
                      <button
                        onClick={() => {
                          setMetadata({ ...metadata, description: metadata.ai_suggested.description });
                          notify('Applied description suggestion', 'success');
                        }}
                        className="text-blue-600 hover:underline text-sm mt-1"
                      >
                        Apply
                      </button>
                    </div>
                  )}
                  
                  <button
                    onClick={() => {
                      const suggestions = metadata.ai_suggested;
                      setMetadata({
                        ...metadata,
                        category: suggestions.category || metadata.category,
                        department: suggestions.department || metadata.department,
                        tags: suggestions.tags ? suggestions.tags.join(', ') : metadata.tags,
                        description: suggestions.description || metadata.description,
                        author: suggestions.author || metadata.author,
                        version: suggestions.version || metadata.version
                      });
                      notify('Applied all AI suggestions', 'success');
                    }}
                    style={{
                      backgroundColor: '#27AE60',
                      color: 'white',
                      padding: '8px 16px',
                      fontSize: '14px',
                      fontWeight: '500',
                      width: '100%',
                      marginTop: '12px'
                    }}
                    className="hover:opacity-90"
                  >
                    ✅ Apply All Suggestions
                  </button>
                </div>
              )}
              
              {!metadata.ai_suggested && (
                <p className="text-sm text-gray-600">
                  Click "Extract Metadata" to get AI-powered suggestions for categories, tags, and other metadata fields.
                </p>
              )}
            </div>
          </div>
        )}

        {activeTab === 'status' && (
          <div className="space-y-6">
            <h2 className="text-lg font-bold mb-4" style={{ color: '#2C3E50' }}>Processing Status</h2>
            
            {/* Document Status */}
            <div style={{ 
              border: '1px solid #E1E8ED',
              padding: '20px',
              marginBottom: '20px'
            }}>
              <h3 className="font-bold mb-4" style={{ color: '#2C3E50' }}>Document Status</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Status:</span>
                  <span className={`font-bold ${
                    document.status === 'ingested' ? 'text-green-600' :
                    document.status === 'processing' ? 'text-blue-600' :
                    document.status === 'failed' ? 'text-red-600' :
                    'text-gray-600'
                  }`}>
                    {document.status.toUpperCase()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Chunks:</span>
                  <span>{document.chunk_count || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Entities:</span>
                  <span>{document.entity_count || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Last Processed:</span>
                  <span>
                    {document.ingested_at ? 
                      new Date(document.ingested_at).toLocaleString() : 
                      'Never'}
                  </span>
                </div>
              </div>
            </div>

            {/* Processing Actions */}
            <div style={{ 
              border: '1px solid #E1E8ED',
              padding: '20px',
              marginBottom: '20px'
            }}>
              <h3 className="font-bold mb-4" style={{ color: '#2C3E50' }}>Processing Actions</h3>
              <div className="space-y-4">
                <button
                  onClick={() => notify('Upgrade parsing tier feature coming soon', 'info')}
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: '#F8F7F3',
                    color: '#95A5A6',
                    border: '1px solid #E1E8ED',
                    fontSize: '14px',
                    cursor: 'not-allowed'
                  }}
                >
                  ⬆️ Upgrade Parsing Tier (Coming Soon)
                </button>
                
                <button
                  onClick={() => notify('Export feature coming soon', 'info')}
                  style={{
                    width: '100%',
                    padding: '10px',
                    backgroundColor: '#F8F7F3',
                    color: '#95A5A6',
                    border: '1px solid #E1E8ED',
                    fontSize: '14px',
                    cursor: 'not-allowed'
                  }}
                >
                  📤 Export to JSON (Coming Soon)
                </button>
              </div>
            </div>

            {/* Processing Metadata */}
            {metadata.processing && (
              <div style={{ 
                border: '1px solid #E1E8ED',
                padding: '20px',
                backgroundColor: '#FAFAFA'
              }}>
                <h3 className="font-bold mb-4" style={{ color: '#7F8C8D' }}>Processing Metadata</h3>
                <div className="space-y-2 text-sm">
                  {Object.entries(metadata.processing).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span>{key}:</span>
                      <span className="font-mono">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}