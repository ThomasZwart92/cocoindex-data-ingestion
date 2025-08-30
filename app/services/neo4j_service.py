"""Neo4j Graph Database Service for storing entities and relationships"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.config import settings
from app.models.relationships import (
    RelationshipType,
    Relationship as RelationshipModel,
    RelationshipProperties
)

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entity to store in graph"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 1.0

@dataclass
class Relationship:
    """Relationship between entities"""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 1.0

class Neo4jService:
    """Service for managing knowledge graph in Neo4j"""
    
    def __init__(self, uri: Optional[str] = None, auth: Optional[Tuple[str, str]] = None):
        """
        Initialize Neo4j service
        
        Args:
            uri: Neo4j connection URI
            auth: (username, password) tuple
        """
        self.uri = uri or settings.neo4j_uri
        username = settings.neo4j_username if auth is None else auth[0]
        password = settings.neo4j_password if auth is None else auth[1]
        
        self.auth = (username, password)
        self.driver: Optional[AsyncDriver] = None
        
        logger.info(f"Initialized Neo4jService with URI: {self.uri}")
    
    async def connect(self):
        """Establish connection to Neo4j"""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=self.auth,
                max_connection_pool_size=50
            )
            
            # Verify connectivity
            try:
                async with self.driver.session() as session:
                    result = await session.run("RETURN 1 as test")
                    await result.single()
                logger.info("Successfully connected to Neo4j")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Closed Neo4j connection")
    
    def execute_query(self, query: str, parameters: dict = None):
        """Execute a Cypher query synchronously"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    async def execute_query_async(self, query: str, parameters: dict = None):
        """Execute a Cypher query asynchronously"""
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return [dict(record) async for record in result]
    
    async def ensure_constraints(self):
        """Create necessary constraints and indexes"""
        await self.connect()
        
        constraints = [
            # Unique constraint on Entity ID
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            # Unique constraint on Document ID
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            # Index on Entity name for faster lookups
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            # Index on Entity type
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            # Composite index for entity resolution
            "CREATE INDEX entity_name_type IF NOT EXISTS FOR (e:Entity) ON (e.name, e.type)"
        ]
        
        async with self.driver.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                    logger.info(f"Created constraint/index: {constraint.split()[2]}")
                except Neo4jError as e:
                    if "already exists" in str(e).lower():
                        logger.debug(f"Constraint/index already exists: {constraint.split()[2]}")
                    else:
                        logger.error(f"Failed to create constraint: {e}")
    
    async def upsert_document(
        self,
        document_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """
        Create or update a document node
        
        Args:
            document_id: Unique document identifier
            properties: Document properties
            
        Returns:
            True if successful
        """
        await self.connect()
        
        query = """
        MERGE (d:Document {id: $document_id})
        SET d += $properties
        SET d.updated_at = datetime()
        RETURN d
        """
        
        # Clean properties for Neo4j
        clean_props = self._clean_properties(properties)
        clean_props["id"] = document_id
        
        async with self.driver.session() as session:
            try:
                result = await session.run(
                    query,
                    document_id=document_id,
                    properties=clean_props
                )
                record = await result.single()
                logger.info(f"Upserted document node: {document_id}")
                return record is not None
            except Exception as e:
                logger.error(f"Failed to upsert document: {e}")
                return False
    
    async def upsert_entities(
        self,
        entities: List[Entity],
        document_id: Optional[str] = None
    ) -> int:
        """
        Create or update entity nodes
        
        Args:
            entities: List of entities to create
            document_id: Optional document to link entities to
            
        Returns:
            Number of entities created/updated
        """
        await self.connect()
        
        # Query for upserting entities
        entity_query = """
        UNWIND $entities as entity
        MERGE (e:Entity {id: entity.id})
        SET e.name = entity.name
        SET e.type = entity.type
        SET e.confidence = entity.confidence
        SET e += entity.properties
        SET e.updated_at = datetime()
        RETURN e
        """
        
        # Query for linking to document
        link_query = """
        MATCH (d:Document {id: $document_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (d)-[r:CONTAINS_ENTITY]->(e)
        SET r.extracted_at = datetime()
        SET r.confidence = $confidence
        RETURN r
        """
        
        # Prepare entity data
        entity_data = []
        for entity in entities:
            clean_props = self._clean_properties(entity.properties)
            entity_data.append({
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "confidence": entity.confidence,
                "properties": clean_props
            })
        
        count = 0
        async with self.driver.session() as session:
            try:
                # Create/update entities
                result = await session.run(entity_query, entities=entity_data)
                records = await result.data()
                count = len(records)
                logger.info(f"Upserted {count} entities")
                
                # Link to document if provided
                if document_id:
                    for entity in entities:
                        await session.run(
                            link_query,
                            document_id=document_id,
                            entity_id=entity.id,
                            confidence=entity.confidence
                        )
                    logger.info(f"Linked {count} entities to document {document_id}")
                
            except Exception as e:
                logger.error(f"Failed to upsert entities: {e}")
                raise
        
        return count
    
    async def create_relationships(
        self,
        relationships: List[Relationship]
    ) -> int:
        """
        Create relationships between entities (legacy method)
        
        Args:
            relationships: List of relationships to create
            
        Returns:
            Number of relationships created
        """
        await self.connect()
        
        query = """
        UNWIND $relationships as rel
        MATCH (source:Entity {id: rel.source_id})
        MATCH (target:Entity {id: rel.target_id})
        MERGE (source)-[r:RELATES_TO {type: rel.type}]->(target)
        SET r += rel.properties
        SET r.confidence = rel.confidence
        SET r.created_at = datetime()
        RETURN r
        """
        
        # Prepare relationship data
        rel_data = []
        for rel in relationships:
            clean_props = self._clean_properties(rel.properties)
            rel_data.append({
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "type": rel.type,
                "confidence": rel.confidence,
                "properties": clean_props
            })
        
        async with self.driver.session() as session:
            try:
                result = await session.run(query, relationships=rel_data)
                records = await result.data()
                count = len(records)
                logger.info(f"Created {count} relationships")
                return count
            except Exception as e:
                logger.error(f"Failed to create relationships: {e}")
                return 0
    
    async def create_typed_relationships(
        self,
        relationships: List[RelationshipModel]
    ) -> int:
        """
        Create strongly-typed relationships between entities
        
        Args:
            relationships: List of RelationshipModel objects
            
        Returns:
            Number of relationships created
        """
        await self.connect()
        
        # Group relationships by type for efficient batch processing
        relationships_by_type = {}
        for rel in relationships:
            rel_type = rel.relationship_type.label
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel)
        
        total_created = 0
        
        async with self.driver.session() as session:
            for rel_type_label, rels in relationships_by_type.items():
                # Create dynamic query for each relationship type
                query = f"""
                UNWIND $relationships as rel
                MATCH (source {{name: rel.source_entity}})
                MATCH (target {{name: rel.target_entity}})
                MERGE (source)-[r:{rel_type_label}]->(target)
                SET r += rel.properties
                SET r.updated_at = datetime()
                RETURN r
                """
                
                # Prepare batch data
                batch_data = []
                for rel in rels:
                    props = rel.to_cypher_properties()
                    batch_data.append({
                        "source_entity": rel.source_entity,
                        "target_entity": rel.target_entity,
                        "properties": props
                    })
                
                try:
                    # Use UNWIND for batch processing (fixes N+1 problem)
                    result = await session.run(query, relationships=batch_data)
                    records = await result.data()
                    count = len(records)
                    total_created += count
                    logger.info(f"Created {count} {rel_type_label} relationships")
                except Exception as e:
                    logger.error(f"Failed to create {rel_type_label} relationships: {e}")
        
        return total_created
    
    async def resolve_entities(
        self,
        threshold: float = 0.9
    ) -> int:
        """
        Merge duplicate entities based on name and type similarity
        
        Args:
            threshold: Similarity threshold for merging
            
        Returns:
            Number of entities merged
        """
        await self.connect()
        
        # Find potential duplicates
        find_duplicates_query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.id < e2.id 
        AND e1.type = e2.type
        AND e1.name = e2.name
        RETURN e1.id as id1, e2.id as id2, e1.name as name, e1.type as type
        """
        
        # Merge entities
        merge_query = """
        MATCH (e1:Entity {id: $id1})
        MATCH (e2:Entity {id: $id2})
        // Transfer all relationships from e2 to e1
        CALL {
            WITH e1, e2
            MATCH (e2)-[r]->(n)
            MERGE (e1)-[r2:RELATES_TO]->(n)
            SET r2 += properties(r)
            DELETE r
            RETURN count(r) as rels_out
        }
        CALL {
            WITH e1, e2
            MATCH (n)-[r]->(e2)
            MERGE (n)-[r2:RELATES_TO]->(e1)
            SET r2 += properties(r)
            DELETE r
            RETURN count(r) as rels_in
        }
        // Merge properties (e1 takes precedence)
        SET e1 += properties(e2)
        DELETE e2
        RETURN e1
        """
        
        merged_count = 0
        async with self.driver.session() as session:
            try:
                # Find duplicates
                result = await session.run(find_duplicates_query)
                duplicates = await result.data()
                
                # Merge each duplicate pair
                for dup in duplicates:
                    merge_result = await session.run(
                        merge_query,
                        id1=dup["id1"],
                        id2=dup["id2"]
                    )
                    if await merge_result.single():
                        merged_count += 1
                        logger.info(f"Merged entities: {dup['name']} ({dup['type']})")
                
                logger.info(f"Resolved {merged_count} duplicate entities")
                return merged_count
                
            except Exception as e:
                logger.error(f"Failed to resolve entities: {e}")
                return 0
    
    async def get_entity_graph(
        self,
        entity_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get entity and its relationships up to specified depth
        
        Args:
            entity_id: Entity ID to start from
            depth: How many hops to traverse
            
        Returns:
            Graph data with nodes and edges
        """
        await self.connect()
        
        query = """
        MATCH path = (e:Entity {id: $entity_id})-[*0..""" + str(depth) + """]-()
        WITH collect(distinct nodes(path)) as node_lists,
             collect(distinct relationships(path)) as rel_lists
        UNWIND node_lists as nodes
        UNWIND nodes as node
        WITH collect(distinct node) as all_nodes, rel_lists
        UNWIND rel_lists as rels
        UNWIND rels as rel
        WITH all_nodes, collect(distinct rel) as all_rels
        RETURN all_nodes, all_rels
        """
        
        async with self.driver.session() as session:
            try:
                result = await session.run(query, entity_id=entity_id)
                record = await result.single()
                
                if not record:
                    return {"nodes": [], "edges": []}
                
                # Format nodes
                nodes = []
                for node in record["all_nodes"]:
                    node_data = dict(node)
                    node_data["labels"] = list(node.labels)
                    nodes.append(node_data)
                
                # Format edges
                edges = []
                for rel in record["all_rels"]:
                    edges.append({
                        "source": rel.start_node.element_id,
                        "target": rel.end_node.element_id,
                        "type": rel.type,
                        "properties": dict(rel)
                    })
                
                return {"nodes": nodes, "edges": edges}
                
            except Exception as e:
                logger.error(f"Failed to get entity graph: {e}")
                return {"nodes": [], "edges": []}
    
    async def search_entities(
        self,
        name_pattern: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by name pattern or type
        
        Args:
            name_pattern: Name pattern to search (uses CONTAINS)
            entity_type: Entity type filter
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        await self.connect()
        
        # Build query based on filters
        where_clauses = []
        params = {"limit": limit}
        
        if name_pattern:
            where_clauses.append("e.name CONTAINS $name_pattern")
            params["name_pattern"] = name_pattern
        
        if entity_type:
            where_clauses.append("e.type = $entity_type")
            params["entity_type"] = entity_type
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
        MATCH (e:Entity)
        {where_clause}
        RETURN e
        ORDER BY e.name
        LIMIT $limit
        """
        
        async with self.driver.session() as session:
            try:
                result = await session.run(query, **params)
                records = await result.data()
                
                entities = []
                for record in records:
                    entity = dict(record["e"])
                    entities.append(entity)
                
                logger.info(f"Found {len(entities)} entities matching search criteria")
                return entities
                
            except Exception as e:
                logger.error(f"Failed to search entities: {e}")
                return []
    
    async def delete_document_graph(self, document_id: str) -> bool:
        """
        Delete document and optionally its orphaned entities
        
        Args:
            document_id: Document to delete
            
        Returns:
            True if successful
        """
        await self.connect()
        
        # Delete document and relationships, but keep entities that have other connections
        query = """
        MATCH (d:Document {id: $document_id})
        OPTIONAL MATCH (d)-[r:CONTAINS_ENTITY]->(e:Entity)
        DELETE r, d
        WITH e
        WHERE e IS NOT NULL
        AND NOT EXISTS {
            MATCH (other:Document)-[:CONTAINS_ENTITY]->(e)
            WHERE other.id <> $document_id
        }
        DELETE e
        RETURN count(e) as deleted_entities
        """
        
        async with self.driver.session() as session:
            try:
                result = await session.run(query, document_id=document_id)
                record = await result.single()
                deleted = record["deleted_entities"] if record else 0
                logger.info(f"Deleted document {document_id} and {deleted} orphaned entities")
                return True
            except Exception as e:
                logger.error(f"Failed to delete document graph: {e}")
                return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        await self.connect()
        
        query = """
        MATCH (d:Document) WITH count(d) as doc_count
        MATCH (e:Entity) WITH doc_count, count(e) as entity_count, 
                              count(distinct e.type) as type_count
        MATCH ()-[r]->() WITH doc_count, entity_count, type_count, 
                             count(r) as rel_count
        RETURN doc_count, entity_count, type_count, rel_count
        """
        
        async with self.driver.session() as session:
            try:
                result = await session.run(query)
                record = await result.single()
                
                if record:
                    return {
                        "documents": record["doc_count"],
                        "entities": record["entity_count"],
                        "entity_types": record["type_count"],
                        "relationships": record["rel_count"]
                    }
                else:
                    return {
                        "documents": 0,
                        "entities": 0,
                        "entity_types": 0,
                        "relationships": 0
                    }
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")
                return {}
    
    async def health_check(self) -> bool:
        """Check if Neo4j is healthy and accessible"""
        try:
            await self.connect()
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
            logger.info("Neo4j is healthy")
            return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
    
    def _clean_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean properties for Neo4j compatibility
        
        Neo4j doesn't support nested objects or lists of objects
        """
        clean = {}
        for key, value in properties.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, datetime):
                clean[key] = value.isoformat()
            elif isinstance(value, (list, tuple)):
                # Only keep lists of primitives
                if all(isinstance(v, (str, int, float, bool)) for v in value):
                    clean[key] = list(value)
                else:
                    clean[key] = json.dumps(value)
            elif isinstance(value, dict):
                # Convert nested dicts to JSON string
                clean[key] = json.dumps(value)
            else:
                # Convert everything else to string
                clean[key] = str(value)
        
        return clean