'use client';

import { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentApi } from '@/lib/api';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useNotification } from '@/components/NotificationProvider';
import ChunkViewer from '@/components/document/ChunkViewer';
import EntitiesRelationships from '@/components/document/EntitiesRelationships';

type TabType = 'preview' | 'entities' | 'metadata' | 'status';

interface ProcessingProgress {
  step: string;
  percentage: number;
  message: string;
  elapsed_time: number;
  status?: string;
  jobId?: string;
  timestamp: number;
}

type ConnectionState = 'idle' | 'connecting' | 'open' | 'error' | 'timeout';

export default function DocumentDetailPage() {
  const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || '').replace(/\/+$/, '');

  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const documentId = params.id as string;
  const { notify, confirm } = useNotification();

  const [activeTab, setActiveTab] = useState<TabType>('preview');
  const [metadata, setMetadata] = useState<Record<string, any>>({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress | null>(null);
  const [progressLog, setProgressLog] = useState<ProcessingProgress[]>([]);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle');
  const [now, setNow] = useState(() => Date.now());

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldListenRef = useRef(false);
  const previousStatusRef = useRef<string | undefined>(undefined);
  const startTimeRef = useRef<number | null>(null);
  const processingProgressRef = useRef<ProcessingProgress | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch document
  const { data: document, isLoading: docLoading } = useQuery({
    queryKey: ['document', documentId],
    queryFn: async () => {
      const response = await documentApi.get(documentId);
      const initialMetadata = response.data.metadata || {};
      // Include security_level in metadata if not already present
      if (!initialMetadata.security_level && response.data.security_level) {
        initialMetadata.security_level = response.data.security_level;
      }
      setMetadata(initialMetadata);
      return response.data;
    },
    enabled: !!documentId,
    refetchInterval: isProcessing ? 4000 : undefined,
    refetchIntervalInBackground: true,
  });

  // Fetch chunks
  const { data: chunks, isLoading: chunksLoading } = useQuery({
    queryKey: ['chunks', documentId],
    queryFn: async () => {
      const response = await documentApi.getChunks(documentId);
      return response.data;
    },
    enabled: !!documentId,
    refetchInterval: isProcessing ? 6000 : undefined,
    refetchOnWindowFocus: !isProcessing,
  });

  // Fetch entities
  const secondsSinceUpdate = lastUpdate ? Math.floor((now - lastUpdate) / 1000) : null;

  const connectionStatusLabel = (() => {
    switch (connectionState) {
      case 'connecting':
        return 'Connecting';
      case 'open':
        return 'Live';
      case 'error':
        return 'Reconnecting';
      case 'timeout':
        return 'No updates';
      default:
        return 'Idle';
    }
  })();

  const connectionStatusColor =
    connectionState === 'open'
      ? '#27AE60'
      : connectionState === 'error'
      ? '#E67E22'
      : connectionState === 'timeout'
      ? '#C0392B'
      : connectionState === 'connecting'
      ? '#3498DB'
      : '#95A5A6';

  const lastUpdateSummary = secondsSinceUpdate !== null
    ? `Last update ${secondsSinceUpdate}s ago`
    : 'Waiting for first update';

  const formatTimestamp = (value: number) => new Date(value).toLocaleTimeString();

  const { data: entities } = useQuery({
    queryKey: ['entities', documentId],
    queryFn: async () => {
      const response = await documentApi.getEntities(documentId);
      return response.data;
    },
    enabled: !!documentId,
    refetchInterval: isProcessing ? 6000 : undefined,
    refetchOnWindowFocus: !isProcessing,
  });

  // Check if document is already processing on load
  useEffect(() => {
    const previousStatus = previousStatusRef.current;
    const currentStatus = document?.status;

    if (currentStatus === 'processing' && previousStatus !== 'processing') {
      startTimeRef.current = Date.now();
      setIsProcessing(true);
      setProgressLog([]);
      setProcessingProgress(null);
      setLastUpdate(null);
    }

    if (currentStatus && currentStatus !== 'processing' && previousStatus === 'processing') {
      setIsProcessing(false);
      setProcessingProgress(null);
    }

    previousStatusRef.current = currentStatus;
  }, [document?.status]);

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    processingProgressRef.current = processingProgress;
  }, [processingProgress]);

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, []);

  // Connect to SSE for progress updates when processing
  useEffect(() => {
    const shouldListen = (isProcessing || document?.status === 'processing') && !!documentId;
    shouldListenRef.current = shouldListen;

    if (!shouldListen) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      setConnectionState('idle');
      return;
    }

    setConnectionState((state) => (state === 'open' ? state : 'connecting'));

    const connect = () => {
      if (!shouldListenRef.current) {
        return;
      }

      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const progressUrl = API_BASE_URL
        ? `${API_BASE_URL}/api/documents/${documentId}/progress?ts=${Date.now()}`
        : `/api/documents/${documentId}/progress?ts=${Date.now()}`;
      const source = new EventSource(progressUrl, { withCredentials: true });
      eventSourceRef.current = source;

      source.onopen = () => {
        setConnectionState('open');
      };

      source.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const timestamp = Date.now();
          setLastUpdate(timestamp);

          if (data.event === 'connected') {
            setConnectionState('open');
            return;
          }

          if (data.event === 'heartbeat') {
            return;
          }

          if (data.event === 'job_update') {
            const percentage = typeof data.progress === 'number'
              ? data.progress
              : (processingProgressRef.current?.percentage ?? 0);
            const elapsed = typeof data.elapsed_time === 'number'
              ? data.elapsed_time
              : (startTimeRef.current
                ? (timestamp - startTimeRef.current) / 1000
                : processingProgressRef.current?.elapsed_time ?? 0);
            const message = data.current_step || `Job status: ${data.status ?? 'update'}`;
            const entry: ProcessingProgress = {
              step: data.current_step || data.status || 'job_update',
              message,
              percentage,
              elapsed_time: elapsed,
              status: data.status,
              jobId: data.job_id,
              timestamp,
            };
            processingProgressRef.current = entry;
            setProcessingProgress(entry);
            setProgressLog((prev) => [entry, ...prev].slice(0, 30));
            return;
          }

          if (data.event === 'timeout') {
            setConnectionState('timeout');
            const elapsed = startTimeRef.current
              ? (timestamp - startTimeRef.current) / 1000
              : processingProgressRef.current?.elapsed_time ?? 0;
            const entry: ProcessingProgress = {
              step: 'timeout',
              message: data.message || 'No updates received for 30 seconds',
              percentage: processingProgressRef.current?.percentage ?? 0,
              elapsed_time: elapsed,
              status: 'timeout',
              timestamp,
            };
            processingProgressRef.current = entry;
            setProcessingProgress(entry);
            setProgressLog((prev) => [entry, ...prev].slice(0, 30));
            return;
          }

          if (data.event === 'error') {
            setConnectionState('error');
            return;
          }

          if (data.event === 'complete') {
            const elapsed = startTimeRef.current
              ? (timestamp - startTimeRef.current) / 1000
              : processingProgressRef.current?.elapsed_time ?? 0;
            const status = data.final_status || 'completed';
            const completionEntry: ProcessingProgress = {
              step: 'complete',
              message: `Processing ${status}`,
              percentage: 100,
              elapsed_time: elapsed,
              status,
              timestamp,
            };
            setProgressLog((prev) => [completionEntry, ...prev].slice(0, 30));
            processingProgressRef.current = null;
            setProcessingProgress(null);
            setIsProcessing(false);
            setConnectionState('idle');
            queryClient.invalidateQueries({ queryKey: ['document', documentId] });
            queryClient.invalidateQueries({ queryKey: ['chunks', documentId] });
            queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
            source.close();
            if (eventSourceRef.current === source) {
              eventSourceRef.current = null;
            }
            if (status === 'success' || status === 'completed') {
              notify('Document processing completed successfully!', 'success');
            } else if (status === 'failed') {
              notify('Document processing failed. Please try again.', 'error');
            }
            return;
          }

          if (data.step) {
            const percentage = typeof data.percentage === 'number'
              ? data.percentage
              : (processingProgressRef.current?.percentage ?? 0);
            const elapsed = typeof data.elapsed_time === 'number'
              ? data.elapsed_time
              : (startTimeRef.current
                ? (timestamp - startTimeRef.current) / 1000
                : processingProgressRef.current?.elapsed_time ?? 0);
            const entry: ProcessingProgress = {
              step: data.step,
              message: data.message ?? data.step,
              percentage,
              elapsed_time: elapsed,
              status: data.status,
              timestamp,
            };
            processingProgressRef.current = entry;
            setProcessingProgress(entry);
            setProgressLog((prev) => [entry, ...prev].slice(0, 30));
            return;
          }
        } catch (error) {
          console.error('Error parsing SSE data:', error);
        }
      };

      source.onerror = (event) => {
        console.warn('SSE connection error', event);
        setConnectionState('error');
        source.close();
        if (eventSourceRef.current === source) {
          eventSourceRef.current = null;
        }
        if (shouldListenRef.current) {
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
          reconnectTimeoutRef.current = setTimeout(connect, 3000);
        }
      };
    };

    connect();

    return () => {
      shouldListenRef.current = false;
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [isProcessing, document?.status, documentId, notify, queryClient]);

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
    mutationFn: async () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }

      setIsProcessing(true);
      startTimeRef.current = Date.now();
      setProgressLog([]);
      setProcessingProgress(null);
      setLastUpdate(null);
      setConnectionState('connecting');
      return documentApi.reprocess(documentId);
    },
    onSuccess: () => {
      notify('Document processing started', 'success');
      pollIntervalRef.current = setInterval(async () => {
        const response = await documentApi.get(documentId);
        if (response.data.status !== 'processing') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          setIsProcessing(false);
          queryClient.invalidateQueries({ queryKey: ['document', documentId] });
          queryClient.invalidateQueries({ queryKey: ['chunks', documentId] });
          queryClient.invalidateQueries({ queryKey: ['entities', documentId] });
          if (response.data.status === 'failed') {
            notify('Document processing failed', 'error');
          } else if (response.data.chunk_count > 0) {
            notify(`Processing complete: ${response.data.chunk_count} chunks created`, 'success');
          }
        }
      }, 4000);

      setTimeout(() => {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      }, 120000);
    },
    onError: (error: any) => {
      setIsProcessing(false);
      setProcessingProgress(null);
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
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
                Source: {document.source_type} | 
                Status: <span style={{
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontWeight: '500',
                  backgroundColor: 
                    document.status === 'pending_review' ? '#FFF3CD' :
                    document.status === 'ingested' ? '#D4EDDA' :
                    document.status === 'processing' ? '#E8F4FD' :
                    document.status === 'failed' ? '#F8D7DA' :
                    document.status === 'rejected' ? '#FFE5E5' :
                    '#F0F0F0',
                  color:
                    document.status === 'pending_review' ? '#856404' :
                    document.status === 'ingested' ? '#155724' :
                    document.status === 'processing' ? '#004085' :
                    document.status === 'failed' ? '#721C24' :
                    document.status === 'rejected' ? '#842029' :
                    '#495057'
                }}>{document.status.toUpperCase()}</span> | 
                Security: {document.security_level} (Level {document.access_level})
              </div>
            </div>
            <div className="flex gap-3">
              {/* Show Approve/Reject buttons for pending_review documents */}
              {document.status === 'pending_review' && (
                <>
                  <button
                    onClick={async () => {
                      const confirmed = await confirm(
                        'Approve this document?\nThis will mark it as reviewed and ready for use.'
                      );
                      if (confirmed) {
                        try {
                          await documentApi.approve(documentId);
                          notify('Document approved successfully', 'success');
                          queryClient.invalidateQueries({ queryKey: ['document', documentId] });
                        } catch (error) {
                          notify('Failed to approve document', 'error');
                        }
                      }
                    }}
                    style={{
                      backgroundColor: '#27AE60',
                      color: 'white',
                      border: 'none',
                      padding: '8px 16px',
                      fontSize: '14px',
                      fontWeight: '500',
                      marginRight: '8px',
                      cursor: 'pointer'
                    }}
                    className="hover:opacity-90"
                  >
                    ✅ Approve
                  </button>
                  <button
                    onClick={async () => {
                      const reason = prompt('Please provide a reason for rejection:');
                      if (reason) {
                        try {
                          await documentApi.reject(documentId, reason);
                          notify('Document rejected', 'warning');
                          queryClient.invalidateQueries({ queryKey: ['document', documentId] });
                        } catch (error) {
                          notify('Failed to reject document', 'error');
                        }
                      }
                    }}
                    style={{
                      backgroundColor: '#E74C3C',
                      color: 'white',
                      border: 'none',
                      padding: '8px 16px',
                      fontSize: '14px',
                      fontWeight: '500',
                      marginRight: '8px',
                      cursor: 'pointer'
                    }}
                    className="hover:opacity-90"
                  >
                    ❌ Reject
                  </button>
                </>
              )}
              
              {/* Process/Reprocess button - hide when pending_review */}
              {document.status !== 'pending_review' && (
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
                      Processing...
                      {processingProgress && (
                        <span className="ml-2 text-xs">({processingProgress.percentage}%)</span>
                      )}
                    </>
                  ) : document.chunk_count > 0 ? (
                    '🔄 Reprocess'
                  ) : (
                    '📊 Process'
                  )}
                </button>
              )}
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

      {/* Processing Banner */}
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
                <div className="space-y-2">
                  {processingProgress ? (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm" style={{ color: '#3498DB' }}>
                          <span>{processingProgress.message}</span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {processingProgress.elapsed_time.toFixed(1)}s
                        </span>
                      </div>
                      <div className="text-xs text-blue-500 mb-2">
                        {lastUpdateSummary}
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${processingProgress.percentage}%` }}
                        />
                      </div>
                      <div className="mt-1 text-xs text-gray-600">
                        Step: {processingProgress.step} ({processingProgress.percentage}%)
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm" style={{ color: '#3498DB' }}>
                      <span>{lastUpdateSummary}</span>
                    </div>
                  )}
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
                  <label className="block text-sm mb-1">Security Level:</label>
                  <select
                    value={metadata.security_level || document.security_level || ''}
                    onChange={(e) => setMetadata({ ...metadata, security_level: e.target.value })}
                    className="w-full"
                  >
                    <option value="">Select security level...</option>
                    <option value="public">Tier 1: Public</option>
                    <option value="client">Tier 2: Client</option>
                    <option value="partner">Tier 3: Partner</option>
                    <option value="employee">Tier 4: Employee</option>
                    <option value="management">Tier 5: Management</option>
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
              border: '1px solid #E1E8ED',
              padding: '24px',
              backgroundColor: '#FAFBFC',
              borderRadius: '4px'
            }}>
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-bold" style={{ color: '#2C3E50' }}>AI Metadata Extraction</h3>
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
                    padding: '8px 16px',
                    fontSize: '14px',
                    fontWeight: '500',
                    borderRadius: '4px',
                    border: 'none',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                  className="hover:opacity-90"
                >
                  <span style={{ fontSize: '16px' }}>🤖</span>
                  Extract Metadata
                </button>
              </div>

              {metadata.ai_suggested ? (
                <div style={{ marginTop: '8px' }}>
                  {/* Category Suggestion */}
                  {metadata.ai_suggested.category && (
                    <div style={{
                      padding: '12px',
                      backgroundColor: 'white',
                      border: '1px solid #E1E8ED',
                      borderRadius: '4px',
                      marginBottom: '12px'
                    }}>
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <span className="text-sm font-semibold" style={{ color: '#5A6C7D' }}>Category:</span>
                          <span className="ml-3 text-sm" style={{ color: '#2C3E50' }}>{metadata.ai_suggested.category}</span>
                        </div>
                        <button
                          onClick={() => {
                            setMetadata({ ...metadata, category: metadata.ai_suggested.category });
                            notify('Applied category suggestion', 'success');
                          }}
                          style={{
                            backgroundColor: '#E8F5FF',
                            color: '#3498DB',
                            padding: '4px 12px',
                            fontSize: '13px',
                            fontWeight: '500',
                            borderRadius: '3px',
                            border: '1px solid #BDD7ED',
                            cursor: 'pointer'
                          }}
                          className="hover:opacity-90"
                        >
                          Apply
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Department Suggestion */}
                  {metadata.ai_suggested.department && (
                    <div style={{
                      padding: '12px',
                      backgroundColor: 'white',
                      border: '1px solid #E1E8ED',
                      borderRadius: '4px',
                      marginBottom: '12px'
                    }}>
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <span className="text-sm font-semibold" style={{ color: '#5A6C7D' }}>Department:</span>
                          <span className="ml-3 text-sm" style={{ color: '#2C3E50' }}>{metadata.ai_suggested.department}</span>
                        </div>
                        <button
                          onClick={() => {
                            setMetadata({ ...metadata, department: metadata.ai_suggested.department });
                            notify('Applied department suggestion', 'success');
                          }}
                          style={{
                            backgroundColor: '#E8F5FF',
                            color: '#3498DB',
                            padding: '4px 12px',
                            fontSize: '13px',
                            fontWeight: '500',
                            borderRadius: '3px',
                            border: '1px solid #BDD7ED',
                            cursor: 'pointer'
                          }}
                          className="hover:opacity-90"
                        >
                          Apply
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Tags Suggestion */}
                  {metadata.ai_suggested.tags && metadata.ai_suggested.tags.length > 0 && (
                    <div style={{
                      padding: '12px',
                      backgroundColor: 'white',
                      border: '1px solid #E1E8ED',
                      borderRadius: '4px',
                      marginBottom: '12px'
                    }}>
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <span className="text-sm font-semibold" style={{ color: '#5A6C7D' }}>Tags:</span>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {metadata.ai_suggested.tags.map((tag: string, index: number) => (
                              <span
                                key={`${tag}-${index}`}
                                style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  padding: '4px 12px',
                                  fontSize: '12px',
                                  backgroundColor: '#E8F5FF',
                                  color: '#3498DB',
                                  borderRadius: '12px',
                                  border: '1px solid #BDD7ED',
                                  marginRight: '4px',
                                  marginBottom: '4px'
                                }}
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>
                        <button
                          onClick={() => {
                            setMetadata({
                              ...metadata,
                              tags: metadata.ai_suggested.tags.join(', ')
                            });
                            notify('Applied tag suggestions', 'success');
                          }}
                          style={{
                            backgroundColor: '#E8F5FF',
                            color: '#3498DB',
                            padding: '4px 12px',
                            fontSize: '13px',
                            fontWeight: '500',
                            borderRadius: '3px',
                            border: '1px solid #BDD7ED',
                            cursor: 'pointer',
                            marginLeft: '12px'
                          }}
                          className="hover:opacity-90"
                        >
                          Apply
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Description Suggestion */}
                  {metadata.ai_suggested.description && (
                    <div style={{
                      padding: '12px',
                      backgroundColor: 'white',
                      border: '1px solid #E1E8ED',
                      borderRadius: '4px',
                      marginBottom: '12px'
                    }}>
                      <div className="flex flex-col">
                        <div className="flex items-start justify-between mb-2">
                          <span className="text-sm font-semibold" style={{ color: '#5A6C7D' }}>Description:</span>
                          <button
                            onClick={() => {
                              setMetadata({ ...metadata, description: metadata.ai_suggested.description });
                              notify('Applied description suggestion', 'success');
                            }}
                            style={{
                              backgroundColor: '#E8F5FF',
                              color: '#3498DB',
                              padding: '4px 12px',
                              fontSize: '13px',
                              fontWeight: '500',
                              borderRadius: '3px',
                              border: '1px solid #BDD7ED',
                              cursor: 'pointer'
                            }}
                            className="hover:opacity-90"
                          >
                            Apply
                          </button>
                        </div>
                        <p className="text-sm" style={{ color: '#2C3E50', lineHeight: '1.5' }}>
                          {metadata.ai_suggested.description}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Apply All Button */}
                  <div style={{
                    marginTop: '20px',
                    paddingTop: '20px',
                    borderTop: '1px solid #E1E8ED'
                  }}>
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
                        padding: '10px 20px',
                        fontSize: '14px',
                        fontWeight: '500',
                        width: '100%',
                        borderRadius: '4px',
                        border: 'none',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px'
                      }}
                      className="hover:opacity-90"
                    >
                      <span style={{ fontSize: '16px' }}>✅</span>
                      Apply All Suggestions
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{
                  padding: '40px',
                  textAlign: 'center',
                  color: '#95A3B3'
                }}>
                  <p className="text-sm">No AI suggestions available yet.</p>
                  <p className="text-xs mt-2">Click "Extract Metadata" to generate suggestions.</p>
                </div>
              )}
            </div>

            {metadata.ai_extracted && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
                  <p className="text-sm text-green-800">
                    ✅ Metadata was automatically extracted during document processing
                  </p>
                  {metadata.extraction_timestamp && (
                    <p className="text-xs text-green-600 mt-1">
                      Extracted at: {new Date(metadata.extraction_timestamp).toLocaleString()}
                    </p>
                  )}
                </div>
              )}
          </div>
        )}

        {activeTab === 'status' && (
          <div className="space-y-6">
            <h2 className="text-lg font-bold mb-4" style={{ color: '#2C3E50' }}>Processing Status</h2>
            
            <div
              style={{
                border: '1px solid #E1E8ED',
                padding: '20px',
                marginBottom: '20px'
              }}
            >
              <h3 className="font-bold mb-4" style={{ color: '#2C3E50' }}>Live Progress</h3>
              <div className="flex justify-between text-sm mb-3">
                <span>
                  Connection:
                  <span style={{ color: connectionStatusColor, fontWeight: 600 }}> {connectionStatusLabel}</span>
                </span>
                <span>
                  Last update: {secondsSinceUpdate !== null ? `${secondsSinceUpdate}s ago` : 'No updates yet'}
                </span>
              </div>
              {progressLog.length > 0 ? (
                <div className="space-y-2 max-h-60 overflow-auto pr-1">
                  {progressLog.slice(0, 20).map((entry, index) => (
                    <div
                      key={`${entry.timestamp}-${index}`}
                      className="border border-blue-100 bg-blue-50/60 rounded p-3"
                    >
                      <div className="flex justify-between">
                        <span className="font-semibold text-blue-900">{entry.message}</span>
                        <span className="text-xs text-blue-700">{entry.percentage}%</span>
                      </div>
                      <div className="text-xs text-blue-600 mt-1">
                        Step: {entry.step}
                        {entry.status ? ` • ${entry.status}` : ''}
                        {' • '}
                        {entry.elapsed_time.toFixed(1)}s
                        {' • '}
                        {formatTimestamp(entry.timestamp)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-blue-700">Waiting for progress updates...</div>
              )}
            </div>

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
                  <span>Metadata:</span>
                  <span className={metadata?.ai_extracted ? 'text-green-600 font-bold' : 'text-gray-600'}>
                    {metadata?.ai_extracted ? 'Extracted' : 'Not extracted'}
                  </span>
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