'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { documentApi, systemApi, processingApi } from '@/lib/api';
import Link from 'next/link';
import { useState } from 'react';
import { useNotification } from '@/components/NotificationProvider';

export default function DashboardPage() {
  const queryClient = useQueryClient();
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [processingDocs, setProcessingDocs] = useState<Set<string>>(new Set());
  const { notify, confirm, prompt } = useNotification();

  // Fetch system health
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const response = await systemApi.health();
      return response.data;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch documents with auto-refresh for processing documents
  const { data: documents, isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: async () => {
      const response = await documentApi.list();
      return response.data;
    },
    refetchInterval: (data) => {
      // Auto-refresh every 2 seconds if any document is processing
      // Ensure data is an array before calling .some()
      const hasProcessing = Array.isArray(data) && data.some((doc: any) => doc.status === 'processing');
      return hasProcessing ? 2000 : false;
    },
  });

  // Delete document mutation
  const deleteMutation = useMutation({
    mutationFn: documentApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  // Process source mutations
  const notionMutation = useMutation({
    mutationFn: processingApi.triggerNotion,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      if (data?.status === 'queued') {
        notify(`NOTION SCAN QUEUED\nJOB: ${data.job_id?.substring(0, 8)}`, 'info');
      } else {
        notify(`NOTION SCAN COMPLETE\nFOUND: ${data?.count || 0} DOCS`, 'success');
      }
    },
    onError: (error) => {
      notify(`NOTION SCAN FAILED`, 'error');
    },
  });

  const gdriveMutation = useMutation({
    mutationFn: processingApi.triggerGDrive,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      if (data?.status === 'queued') {
        notify(`GDRIVE SCAN QUEUED\nJOB: ${data.job_id?.substring(0, 8)}`, 'info');
      } else {
        notify(`GDRIVE SCAN COMPLETE\nFOUND: ${data?.count || 0} DOCS`, 'success');
      }
    },
    onError: (error) => {
      notify(`GDRIVE SCAN FAILED`, 'error');
    },
  });

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      connected: 'status-online',
      disconnected: 'status-offline',
      healthy: 'status-online',
      degraded: 'status-pending',
      unhealthy: 'status-offline',
      pending: 'status-pending',
      processing: 'status-processing',
      approved: 'status-approved',
      failed: 'status-failed',
      ingested: 'status-online',
    };
    return colors[status] || '';
  };

  return (
    <div style={{ maxWidth: '85%', margin: '0 auto', paddingTop: '24px', paddingBottom: '24px' }} className="space-y-6">
      {/* System Status */}
      <section>
        <h2 className="text-lg font-bold mb-4" style={{ color: '#2C3E50' }}>System Status</h2>
        <div className="grid grid-cols-4 gap-4">
          {health?.services && Object.entries(health.services).map(([service, status]) => (
            <div key={service} style={{ 
              border: '1px solid #E1E8ED',
              padding: '16px'
            }}>
              <div style={{ fontWeight: '500', textTransform: 'capitalize', color: '#2C3E50' }}>{service}</div>
              <div className={`text-sm mt-1 ${getStatusColor(status.split(':')[0])}`}>
                • {status.charAt(0).toUpperCase() + status.slice(1).toLowerCase()}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Source Controls */}
      <section>
        <h2 className="text-lg font-bold mb-4" style={{ color: '#2C3E50' }}>Sources</h2>
        <div style={{ 
          border: '1px solid #E1E8ED',
          padding: '20px'
        }} className="space-y-3">
          <div className="flex justify-between items-center">
            <span>Notion: Connected</span>
            <button 
              onClick={async () => {
                const confirmed = await confirm('SCAN NOTION?\n\nWILL CHECK ALL CONNECTED PAGES');
                if (confirmed) {
                  notionMutation.mutate();
                }
              }}
              disabled={notionMutation.isPending}
              style={{
                padding: '6px 16px',
                fontSize: '13px',
                fontWeight: '500'
              }}
            >
              {notionMutation.isPending ? 'SCANNING...' : 'SCAN NOW'}
            </button>
          </div>
          <div className="flex justify-between items-center">
            <span>Google Drive: Connected</span>
            <button 
              onClick={async () => {
                const confirmed = await confirm('SCAN GOOGLE DRIVE?\n\nWILL CHECK ALL CONFIGURED FOLDERS');
                if (confirmed) {
                  gdriveMutation.mutate();
                }
              }}
              disabled={gdriveMutation.isPending}
              style={{
                padding: '6px 16px',
                fontSize: '13px',
                fontWeight: '500'
              }}
            >
              {gdriveMutation.isPending ? 'SCANNING...' : 'SCAN NOW'}
            </button>
          </div>
        </div>
      </section>

      {/* Documents Table */}
      <section>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold" style={{ color: '#2C3E50' }}>Documents</h2>
          <div className="space-x-2">
            <button 
              onClick={async () => {
                const url = await prompt('ENTER DOCUMENT URL:\n\n• NOTION PAGE\n• GOOGLE DRIVE FILE\n• LOCAL PATH');
                if (url) {
                  notify(`NOT IMPLEMENTED\nURL: ${url}`, 'warning');
                }
              }}
              className="px-3 py-1 text-sm"
            >
              + ADD
            </button>
          </div>
        </div>
        
        {isLoading ? (
          <div style={{ 
            border: '1px solid #E1E8ED',
            padding: '32px',
            textAlign: 'center',
            color: '#7F8C8D'
          }}>Loading documents...</div>
        ) : error ? (
          <div style={{ 
            border: '1px solid #FDEDEC',
            padding: '32px',
            textAlign: 'center',
            backgroundColor: '#FFF5F5',
            color: '#E74C3C'
          }}>
            Error loading documents: {(error as any)?.message}
          </div>
        ) : !documents || documents.length === 0 ? (
          <div style={{ 
            border: '1px solid #E1E8ED',
            padding: '32px',
            textAlign: 'center',
            color: '#7F8C8D'
          }}>
            No documents found. Trigger a scan to discover documents.
          </div>
        ) : (
          <div style={{ border: '1px solid #E1E8ED', overflow: 'hidden' }}>
            <table className="data-table">
              <thead style={{ backgroundColor: '#F8F7F3' }}>
                <tr>
                  <th className="text-left p-2 w-8">
                    <input 
                      type="checkbox"
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedDocs(new Set(documents.map(d => d.id)));
                        } else {
                          setSelectedDocs(new Set());
                        }
                      }}
                    />
                  </th>
                  <th className="text-left p-3" style={{ fontWeight: '500', color: '#2C3E50' }}>ID</th>
                  <th className="text-left p-3" style={{ fontWeight: '500', color: '#2C3E50' }}>Source</th>
                  <th className="text-left p-3" style={{ fontWeight: '500', color: '#2C3E50' }}>Title</th>
                  <th className="text-left p-3" style={{ fontWeight: '500', color: '#2C3E50' }}>Status</th>
                  <th className="text-left p-3" style={{ fontWeight: '500', color: '#2C3E50' }}>Created</th>
                  <th className="text-left p-3" style={{ fontWeight: '500', color: '#2C3E50' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id} style={{ borderBottom: '1px solid #F0EDE5' }} className="hover:bg-gray-50">
                    <td className="p-3">
                      <input 
                        type="checkbox"
                        checked={selectedDocs.has(doc.id)}
                        onChange={(e) => {
                          const newSelected = new Set(selectedDocs);
                          if (e.target.checked) {
                            newSelected.add(doc.id);
                          } else {
                            newSelected.delete(doc.id);
                          }
                          setSelectedDocs(newSelected);
                        }}
                      />
                    </td>
                    <td className="p-3" style={{ color: '#7F8C8D', fontSize: '13px' }}>
                      {doc.id.substring(0, 8)}...
                    </td>
                    <td className="p-3" style={{ textTransform: 'uppercase' }}>{doc.source_type}</td>
                    <td className="p-3">
                      <Link 
                        href={`/document/${doc.id}`}
                        className="hover:text-blue-600 underline"
                      >
                        {doc.title}
                      </Link>
                    </td>
                    <td className="p-3">
                      <span 
                        style={{ 
                          color: getStatusColor(doc.status).includes('green') ? '#27AE60' : 
                                 getStatusColor(doc.status).includes('red') ? '#E74C3C' : 
                                 getStatusColor(doc.status).includes('blue') ? '#3498DB' : '#F39C12',
                          backgroundColor: getStatusColor(doc.status).includes('green') ? '#E8F8F5' : 
                                          getStatusColor(doc.status).includes('red') ? '#FDEDEC' : 
                                          getStatusColor(doc.status).includes('blue') ? '#EBF5FB' : '#FEF5E7',
                          padding: '4px 12px',
                          fontSize: '12px',
                          fontWeight: '500'
                        }}
                      >
                        {doc.status.charAt(0).toUpperCase() + doc.status.slice(1)}
                      </span>
                    </td>
                    <td className="p-3" style={{ fontSize: '13px' }}>
                      {new Date(doc.created_at).toLocaleDateString()}
                    </td>
                    <td className="p-3 space-x-2">
                      <Link 
                        href={`/document/${doc.id}`}
                        style={{
                          backgroundColor: '#EBF5FB',
                          color: '#3498DB',
                          padding: '6px 12px',
                          fontSize: '13px',
                          textDecoration: 'none',
                          display: 'inline-block'
                        }}
                      >
                        ✏️ Edit
                      </Link>
                      {(doc.status === 'discovered' || doc.status === 'failed') && (
                        <button 
                          onClick={async () => {
                            const confirmed = await confirm(`Process document "${doc.title}"?\n\nThis will:\n• Chunk the document\n• Extract entities\n• Generate embeddings\n• Store in vector database`);
                            if (confirmed) {
                              try {
                                setProcessingDocs(prev => new Set(prev).add(doc.id));
                                const response = await fetch(`http://localhost:8001/api/documents/${doc.id}/process`, {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' }
                                });
                                if (response.ok) {
                                  notify(`Processing started for "${doc.title}"`, 'success');
                                  queryClient.invalidateQueries({ queryKey: ['documents'] });
                                } else {
                                  notify(`Failed to start processing`, 'error');
                                  setProcessingDocs(prev => {
                                    const next = new Set(prev);
                                    next.delete(doc.id);
                                    return next;
                                  });
                                }
                              } catch (error: any) {
                                notify(`Error: ${error.message}`, 'error');
                                setProcessingDocs(prev => {
                                  const next = new Set(prev);
                                  next.delete(doc.id);
                                  return next;
                                });
                              }
                            }
                          }}
                          disabled={processingDocs.has(doc.id)}
                          style={{
                            backgroundColor: processingDocs.has(doc.id) ? '#E8F8F5' : '#FEF5E7',
                            color: processingDocs.has(doc.id) ? '#27AE60' : '#F39C12',
                            padding: '6px 12px',
                            border: 'none',
                            fontSize: '13px',
                            cursor: processingDocs.has(doc.id) ? 'not-allowed' : 'pointer',
                            opacity: processingDocs.has(doc.id) ? 0.7 : 1
                          }}
                        >
                          {processingDocs.has(doc.id) ? '⏳ Starting...' : '⚙️ Process'}
                        </button>
                      )}
                      {doc.status === 'processing' && (
                        <span style={{
                          backgroundColor: '#EBF5FB',
                          color: '#3498DB',
                          padding: '6px 12px',
                          fontSize: '13px',
                          display: 'inline-block',
                          animation: 'pulse 1.5s infinite'
                        }}>
                          ⏳ Processing...
                        </span>
                      )}
                      <button 
                        onClick={async () => {
                          const confirmed = await confirm(`Delete document "${doc.title}"?\nThis will permanently remove the document and all its chunks, entities, and metadata.`);
                          if (confirmed) {
                            deleteMutation.mutate(doc.id);
                            notify(`Document "${doc.title}" deleted`, 'success');
                          }
                        }}
                        style={{
                          backgroundColor: '#FDEDEC',
                          color: '#E74C3C',
                          padding: '6px 12px',
                          border: 'none',
                          fontSize: '13px'
                        }}
                      >
                        🗑 Delete
                      </button>
                      {doc.status !== 'discovered' && (
                        <button 
                          onClick={async () => {
                            const chunkSize = await prompt(`Rechunk document "${doc.title}"?\nCurrent chunk size: 1500 bytes\nEnter new chunk size (bytes):`, '1500');
                            if (chunkSize && !isNaN(parseInt(chunkSize))) {
                              notify(`Rechunking not yet implemented.\nWould rechunk with size: ${chunkSize} bytes`, 'warning');
                            }
                          }}
                          style={{
                            backgroundColor: '#E8F8F5',
                            color: '#27AE60',
                            padding: '6px 12px',
                            border: 'none',
                            fontSize: '13px'
                          }}
                        >
                          🔄 Rechunk
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            
            {/* Table Footer */}
            <div className="p-2 text-xs text-gray-600 border-t border-black">
              [TAB] select • [E] edit • [D] delete • [R] rechunk
            </div>
          </div>
        )}
      </section>
    </div>
  );
}