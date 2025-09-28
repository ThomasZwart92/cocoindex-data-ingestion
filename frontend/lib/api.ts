import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth (if needed later)
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// API Types
export interface Document {
  id: string;
  source_type: 'notion' | 'google_drive' | 'local';
  source_id: string;
  title: string;
  content?: string;
  metadata?: Record<string, any>;
  mime_type?: string;
  author?: string;
  status: 'discovered' | 'processing' | 'pending_review' | 'approved' | 'failed' | 'ingested';
  security_level?: string;
  access_level?: number;
  chunk_count?: number;
  entity_count?: number;
  created_at: string;
  updated_at: string;
  ingested_at?: string;
}

export interface DocumentWithDetails extends Document {
  chunks?: Chunk[];
  entities?: Entity[];
}

export interface Chunk {
  id: string;
  document_id: string;
  chunk_number: number;
  chunk_text: string;
  chunk_size: number;
  start_position: number;
  end_position: number;
  metadata: Record<string, any>;
}

export interface Entity {
  id: string;
  document_id: string;
  entity_type: string;
  entity_name: string;
  confidence: number;
  metadata: Record<string, any>;
  is_verified?: boolean;
  is_edited?: boolean;
  original_name?: string;
  canonical_entity_id?: string | null;
  canonical_name?: string | null;
  canonical_type?: string | null;
  canonical_metadata?: Record<string, any> | null;
  context?: string;
}

export interface ProcessingJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  source_type: string;
  created_at: string;
  error?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    postgresql: string;
    redis: string;
    neo4j: string;
    qdrant: string;
  };
}

// API Functions
export const documentApi = {
  list: () => api.get<Document[]>('/api/documents/'),
  get: (id: string) => api.get<DocumentWithDetails>(`/api/documents/${id}`),
  update: (id: string, data: Partial<Document>) => 
    api.put<Document>(`/api/documents/${id}`, data),
  delete: (id: string, hardDelete: boolean = true) => 
    api.delete(`/api/documents/${id}?hard_delete=${hardDelete}`),
  reprocess: (id: string) => api.post(`/api/documents/${id}/process`),
  rechunk: (id: string) =>
    api.post(`/api/documents/${id}/process`, {
      force_reprocess: true
    }),
  
  // Chunks
  getChunks: (documentId: string) => 
    api.get<Chunk[]>(`/api/documents/${documentId}/chunks`),
  updateChunk: (chunkId: string, text: string) =>
    api.put<Chunk>(`/api/chunks/${chunkId}`, { chunk_text: text }),
  deleteChunk: (chunkId: string) =>
    api.delete(`/api/chunks/${chunkId}`),
  
  // Entities  
  getEntities: (documentId: string) =>
    api.get<Entity[]>(`/api/documents/${documentId}/entities`),
  updateEntity: (entityId: string, data: Partial<Entity>) =>
    api.put<Entity>(`/api/entities/${entityId}`, data),
  deleteEntity: (entityId: string) =>
    api.delete(`/api/entities/${entityId}`),
  addEntity: (documentId: string, entity: Omit<Entity, 'id'>) =>
    api.post<Entity>(`/api/entities`, { ...entity, document_id: documentId }),
  createEntity: (documentId: string, data: any) =>
    api.post<Entity>(`/api/entities`, { ...data, document_id: documentId }),
  
  // Relationships
  getRelationships: (documentId: string) => 
    api.get<any[]>(`/api/documents/${documentId}/relationships`),
  createRelationship: (documentId: string, data: any) =>
    api.post(`/api/documents/${documentId}/relationships`, data),
  updateRelationship: (relationshipId: string, data: any) =>
    api.put(`/api/relationships/${relationshipId}`, data),
  deleteRelationship: (relationshipId: string) =>
    api.delete(`/api/relationships/${relationshipId}`),
  
  // Duplicate detection and merging
  findDuplicates: (documentId: string, threshold = 0.85, autoMerge = false) =>
    api.post(`/api/entities/find-duplicates`, null, {
      params: { document_id: documentId, threshold, auto_merge: autoMerge }
    }),
  mergeEntities: (entityIds: string[], targetName: string, targetType: string) =>
    api.post(`/api/entities/merge`, {
      entity_ids: entityIds,
      target_name: targetName,
      target_type: targetType
    }),
  
  // Metadata extraction
  extractMetadata: (documentId: string) =>
    api.post(`/api/documents/${documentId}/extract-metadata`),
  getSuggestedMetadata: (documentId: string) =>
    api.get<any>(`/api/documents/${documentId}/suggested-metadata`),
  
  // Review workflow
  approve: (documentId: string) =>
    api.post(`/api/documents/${documentId}/approve`),
  reject: (documentId: string, reason: string) =>
    api.post(`/api/documents/${documentId}/reject?reason=${encodeURIComponent(reason)}`),
  getReviewStatus: (documentId: string) =>
    api.get<any>(`/api/documents/${documentId}/review-status`),
};

export const processingApi = {
  triggerNotion: () => api.post<ProcessingJob>('/api/process/notion'),
  triggerGDrive: () => api.post<ProcessingJob>('/api/process/gdrive'),
  getJobStatus: (jobId: string) => api.get<ProcessingJob>(`/api/jobs/${jobId}`),
};

export const systemApi = {
  health: () => api.get<HealthStatus>('/health'),
};
