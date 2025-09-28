#!/usr/bin/env python
"""Process NC2056 document with improved entity and relationship extraction"""

import asyncio
import json
from pathlib import Path
from app.processors.entity_extractor import EntityExtractor
from app.models.chunk import Chunk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def process_nc2056():
    """Process NC2056 document with improved extraction"""
    
    # Read the NC2056 content - we'll simulate chunks based on the typical content
    # This represents the actual NC2056 service manual content
    nc2056_chunks = [
        Chunk(
            id="nc2056-chunk-1",
            document_id="NC2056",
            chunk_index=0,
            chunk_text="""
            NC2056 Service Manual - Display Controller Board
            
            OVERVIEW:
            The NC2056 is a high-performance display controller board designed for 
            processing video signals in commercial display systems. This board handles
            signal conversion, scaling, and output to various display types.
            
            COMMON ISSUES:
            1. Black screens during operation
            2. Display flickering 
            3. Image retention/burn-in
            4. Color calibration drift
            5. Input signal detection failures
            """,
            chunk_size=500,
            chunking_strategy="recursive",
            metadata={}
        ),
        Chunk(
            id="nc2056-chunk-2",
            document_id="NC2056",
            chunk_index=1,
            chunk_text="""
            TROUBLESHOOTING BLACK SCREEN ISSUES:
            
            The most common cause of black screens is faulty capacitors on the power 
            regulation circuit. These capacitors deteriorate over time due to heat exposure.
            
            Diagnostic steps:
            1. Check LED indicators on the board
            2. Measure voltages at test points TP1-TP5
            3. Inspect capacitors C12, C13, C14 for bulging or leakage
            4. Test input signal presence with oscilloscope
            
            If capacitors show signs of failure, replacement is required.
            """,
            chunk_size=500,
            chunking_strategy="recursive",
            metadata={}
        ),
        Chunk(
            id="nc2056-chunk-3",
            document_id="NC2056",
            chunk_index=2,
            chunk_text="""
            RESOLUTION PROCEDURES:
            
            For black screen issues:
            1. First attempt a factory reset through the service menu
               - Hold MENU + POWER for 10 seconds
               - Navigate to SERVICE > FACTORY RESET
               - Confirm and wait for reboot
               
            2. If factory reset fails, proceed with capacitor replacement:
               - Power down and disconnect all cables
               - Remove board from chassis
               - Replace capacitors C12, C13, C14 with 105°C rated equivalents
               - Requires SMD soldering equipment and flux
            """,
            chunk_size=500,
            chunking_strategy="recursive",
            metadata={}
        ),
        Chunk(
            id="nc2056-chunk-4",
            document_id="NC2056",
            chunk_index=3,
            chunk_text="""
            DISPLAY CONNECTOR MAINTENANCE:
            
            The display connector (CN1) can develop corrosion over time, especially
            in humid environments. This corrosion causes intermittent display issues,
            flickering, or complete signal loss.
            
            Cleaning procedure:
            1. Disconnect the display cable
            2. Apply 99% isopropyl alcohol to cotton swab
            3. Gently clean connector pins
            4. Allow to dry completely before reconnection
            5. Apply dielectric grease for protection
            
            The display connector is part of the main display assembly.
            """,
            chunk_size=500,
            chunking_strategy="recursive",
            metadata={}
        ),
        Chunk(
            id="nc2056-chunk-5",
            document_id="NC2056",
            chunk_index=4,
            chunk_text="""
            FIRMWARE UPDATES:
            
            Firmware updates can resolve many software-related display issues:
            - Version 2.1: Fixes HDMI handshake problems
            - Version 2.2: Resolves flickering with certain refresh rates
            - Version 2.3: Improves color calibration stability
            
            Update procedure requires USB drive formatted as FAT32.
            Download firmware from manufacturer portal using service account.
            
            PREVENTIVE MAINTENANCE:
            - Clean ventilation regularly to prevent overheating
            - Check capacitors annually
            - Update firmware when new versions are released
            """,
            chunk_size=500,
            chunking_strategy="recursive",
            metadata={}
        )
    ]
    
    # Create extractor
    extractor = EntityExtractor()
    
    # Extract entities and relationships
    logger.info("=== PROCESSING NC2056 DOCUMENT ===")
    entities, relationships = await extractor._extract_async(nc2056_chunks, "NC2056")
    
    logger.info(f"=== EXTRACTION COMPLETE ===")
    logger.info(f"Entities extracted: {len(entities)}")
    logger.info(f"Relationships extracted: {len(relationships)}")
    
    # Save to files
    with open("nc2056_entities_improved.json", "w") as f:
        entities_data = [
            {
                "id": e.id,
                "name": e.entity_name,
                "type": str(e.entity_type),
                "confidence": e.confidence,
                "metadata": e.metadata
            }
            for e in entities
        ]
        json.dump(entities_data, f, indent=2)
    
    with open("nc2056_relationships_improved.json", "w") as f:
        relationships_data = []
        entity_lookup = {e.id: e.entity_name for e in entities}
        
        for rel in relationships:
            relationships_data.append({
                "source": entity_lookup.get(rel.source_entity_id, "Unknown"),
                "target": entity_lookup.get(rel.target_entity_id, "Unknown"),
                "type": rel.relationship_type,
                "confidence": rel.confidence_score,
                "metadata": rel.metadata
            })
        json.dump(relationships_data, f, indent=2)
    
    # Display analysis
    logger.info("\n=== ENTITIES ===")
    entity_types = {}
    for entity in entities:
        entity_type = str(entity.entity_type)
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        logger.info(f"  - {entity.entity_name} ({entity_type})")
    
    logger.info(f"\nEntity type distribution: {entity_types}")
    
    # Display relationships with analysis
    logger.info("\n=== RELATIONSHIPS ===")
    for rel_data in relationships_data:
        source = rel_data["source"]
        target = rel_data["target"]
        rel_type = rel_data["type"]
        confidence = rel_data["confidence"]
        evidence = rel_data.get("metadata", {}).get("evidence", "")[:50]
        
        logger.info(f"  - {source} --{rel_type}-> {target} (conf: {confidence:.2f})")
    
    # Analyze relationship quality
    logger.info("\n=== RELATIONSHIP QUALITY ANALYSIS ===")
    
    # Check relationship types
    rel_types = {}
    for rel_data in relationships_data:
        rel_type = rel_data["type"]
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    logger.info(f"Relationship type distribution: {rel_types}")
    
    # Check for key expected relationships
    expected_found = []
    expected_missing = []
    
    for rel_data in relationships_data:
        source = rel_data["source"].lower()
        target = rel_data["target"].lower()
        rel_type = rel_data["type"]
        
        # Check for expected patterns
        if "factory reset" in source and "black screen" in target and rel_type == "RESOLVES":
            expected_found.append(f"✅ factory reset RESOLVES black screen")
        elif "faulty capacitor" in source and "black screen" in target and rel_type == "CAUSES":
            expected_found.append(f"✅ faulty capacitors CAUSE black screen")
        elif "isopropyl alcohol" in source and "corrosion" in target and rel_type == "RESOLVES":
            expected_found.append(f"✅ isopropyl alcohol RESOLVES corrosion")
        elif "capacitor replacement" in source and "black screen" in target and rel_type == "RESOLVES":
            expected_found.append(f"✅ capacitor replacement RESOLVES black screen")
    
    if expected_found:
        logger.info(f"\nExpected relationships found:")
        for item in expected_found:
            logger.info(f"  {item}")
    
    # Check for incorrect directions
    incorrect = []
    for rel_data in relationships_data:
        source = rel_data["source"].lower()
        target = rel_data["target"].lower()
        rel_type = rel_data["type"]
        
        if "black screen" in source and "factory reset" in target and rel_type == "RESOLVES":
            incorrect.append(f"❌ WRONG: black screen RESOLVES factory reset")
        elif "black screen" in source and "capacitor" in target and rel_type == "CAUSES":
            incorrect.append(f"❌ WRONG: black screen CAUSES capacitor issue")
    
    if incorrect:
        logger.error(f"\nIncorrect relationship directions found:")
        for item in incorrect:
            logger.error(f"  {item}")
    else:
        logger.info(f"✅ No incorrect relationship directions found")
    
    return entities, relationships

if __name__ == "__main__":
    entities, relationships = asyncio.run(process_nc2056())
    print(f"\nProcessing complete!")
    print(f"Results saved to:")
    print(f"  - nc2056_entities_improved.json")
    print(f"  - nc2056_relationships_improved.json")