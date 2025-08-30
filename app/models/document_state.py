"""
Document State Machine Model
Manages document lifecycle states and transitions
"""
from enum import Enum
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field


class DocumentState(Enum):
    """Document processing states"""
    DISCOVERED = "discovered"           # Document found in source
    PROCESSING = "processing"           # Being processed by CocoIndex
    PENDING_REVIEW = "pending_review"   # Awaiting human review
    APPROVED = "approved"               # Approved for ingestion
    INGESTED = "ingested"              # Successfully stored in databases
    FAILED = "failed"                  # Processing failed
    REJECTED = "rejected"              # Rejected during review
    
    @classmethod
    def valid_transitions(cls) -> Dict[str, List[str]]:
        """Define valid state transitions"""
        return {
            cls.DISCOVERED: [cls.PROCESSING, cls.FAILED],
            cls.PROCESSING: [cls.PENDING_REVIEW, cls.FAILED],
            cls.PENDING_REVIEW: [cls.APPROVED, cls.REJECTED, cls.FAILED],
            cls.APPROVED: [cls.INGESTED, cls.FAILED],
            cls.INGESTED: [],  # Terminal state
            cls.FAILED: [cls.DISCOVERED, cls.PROCESSING],  # Can retry
            cls.REJECTED: [cls.DISCOVERED],  # Can reprocess
        }
    
    def can_transition_to(self, target_state: 'DocumentState') -> bool:
        """Check if transition to target state is valid"""
        valid_states = self.valid_transitions().get(self, [])
        return target_state in valid_states
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state"""
        return self in [self.INGESTED]
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error state"""
        return self in [self.FAILED, self.REJECTED]
    
    @property
    def requires_action(self) -> bool:
        """Check if state requires user action"""
        return self in [self.PENDING_REVIEW, self.FAILED, self.REJECTED]


@dataclass
class StateTransition:
    """Record of a state transition"""
    from_state: DocumentState
    to_state: DocumentState
    timestamp: datetime
    user_id: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "reason": self.reason,
            "metadata": self.metadata
        }


@dataclass
class DocumentStateInfo:
    """Complete document state information"""
    document_id: str
    current_state: DocumentState
    created_at: datetime
    updated_at: datetime
    transition_history: List[StateTransition] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_transition(self, 
                      to_state: DocumentState,
                      user_id: Optional[str] = None,
                      reason: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> StateTransition:
        """Add a state transition"""
        if not self.current_state.can_transition_to(to_state):
            raise ValueError(
                f"Invalid transition from {self.current_state.value} to {to_state.value}"
            )
        
        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            reason=reason,
            metadata=metadata or {}
        )
        
        self.transition_history.append(transition)
        self.current_state = to_state
        self.updated_at = transition.timestamp
        
        # Update counters
        if to_state == DocumentState.FAILED:
            self.error_count += 1
        if to_state == DocumentState.PROCESSING and self.current_state == DocumentState.FAILED:
            self.retry_count += 1
            
        return transition
    
    def get_time_in_state(self) -> float:
        """Get time spent in current state (seconds)"""
        if self.transition_history:
            last_transition = self.transition_history[-1]
            delta = datetime.utcnow() - last_transition.timestamp
            return delta.total_seconds()
        else:
            delta = datetime.utcnow() - self.created_at
            return delta.total_seconds()
    
    def get_processing_duration(self) -> Optional[float]:
        """Get total processing time if completed (seconds)"""
        if self.current_state != DocumentState.INGESTED:
            return None
            
        # Find when processing started
        start_time = None
        end_time = None
        
        for transition in self.transition_history:
            if transition.to_state == DocumentState.PROCESSING and start_time is None:
                start_time = transition.timestamp
            if transition.to_state == DocumentState.INGESTED:
                end_time = transition.timestamp
                
        if start_time and end_time:
            delta = end_time - start_time
            return delta.total_seconds()
            
        return None
    
    def get_state_durations(self) -> Dict[str, float]:
        """Get time spent in each state (seconds)"""
        durations = {}
        
        if not self.transition_history:
            # Still in initial state
            durations[self.current_state.value] = self.get_time_in_state()
            return durations
            
        # Calculate time in each state
        for i, transition in enumerate(self.transition_history):
            if i == 0:
                # Time from creation to first transition
                delta = transition.timestamp - self.created_at
                durations[transition.from_state.value] = delta.total_seconds()
            else:
                # Time between transitions
                prev_transition = self.transition_history[i-1]
                delta = transition.timestamp - prev_transition.timestamp
                state = transition.from_state.value
                durations[state] = durations.get(state, 0) + delta.total_seconds()
        
        # Time in current state
        if self.transition_history:
            last_transition = self.transition_history[-1]
            delta = datetime.utcnow() - last_transition.timestamp
            state = self.current_state.value
            durations[state] = durations.get(state, 0) + delta.total_seconds()
            
        return durations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "document_id": self.document_id,
            "current_state": self.current_state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "transition_history": [t.to_dict() for t in self.transition_history],
            "error_count": self.error_count,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "time_in_state": self.get_time_in_state(),
            "state_durations": self.get_state_durations()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentStateInfo':
        """Create from dictionary"""
        transitions = []
        for t_data in data.get("transition_history", []):
            transitions.append(StateTransition(
                from_state=DocumentState(t_data["from_state"]),
                to_state=DocumentState(t_data["to_state"]),
                timestamp=datetime.fromisoformat(t_data["timestamp"]),
                user_id=t_data.get("user_id"),
                reason=t_data.get("reason"),
                metadata=t_data.get("metadata", {})
            ))
            
        return cls(
            document_id=data["document_id"],
            current_state=DocumentState(data["current_state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            transition_history=transitions,
            error_count=data.get("error_count", 0),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {})
        )


class StateValidationError(Exception):
    """Raised when state transition is invalid"""
    pass


class StateManager:
    """Base interface for state management"""
    
    async def get_state(self, document_id: str) -> Optional[DocumentStateInfo]:
        """Get current state for document"""
        raise NotImplementedError
    
    async def create_state(self, document_id: str, 
                          initial_state: DocumentState = DocumentState.DISCOVERED,
                          metadata: Optional[Dict[str, Any]] = None) -> DocumentStateInfo:
        """Create initial state for document"""
        raise NotImplementedError
    
    async def transition(self, document_id: str,
                        to_state: DocumentState,
                        user_id: Optional[str] = None,
                        reason: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> DocumentStateInfo:
        """Transition document to new state"""
        raise NotImplementedError
    
    async def get_documents_by_state(self, state: DocumentState,
                                    limit: int = 100) -> List[DocumentStateInfo]:
        """Get all documents in a specific state"""
        raise NotImplementedError
    
    async def get_documents_requiring_action(self, 
                                            limit: int = 100) -> List[DocumentStateInfo]:
        """Get documents that require user action"""
        raise NotImplementedError
    
    async def get_state_statistics(self) -> Dict[str, int]:
        """Get count of documents in each state"""
        raise NotImplementedError