"""
Process existing documents with two-tier chunking and save to database.
"""

import asyncio
from app.processors.two_tier_chunker import TwoTierChunker
from app.services.supabase_service import SupabaseService
from app.connectors.notion_connector import NotionConnector


async def process_nc2050():
    """Process NC2050 document with two-tier chunking."""
    
    # Initialize services
    chunker = TwoTierChunker()
    supabase = SupabaseService()
    
    # Get the NC2050 document
    document_id = "e86cbf9f-5fb7-4f5d-8ff9-44633a432a2d"
    
    print("Fetching NC2050 document...")
    doc_response = supabase.client.table('documents').select('*').eq('id', document_id).execute()
    
    if not doc_response.data:
        print("Document not found!")
        return
    
    document = doc_response.data[0]
    content = document.get('content', '')
    title = document.get('title', 'NC2050')
    
    print(f"\nDocument: {title}")
    print(f"Content length: {len(content)} characters")
    print("\n" + "="*80)
    
    # Clear existing chunks
    print("Clearing existing chunks...")
    supabase.client.table('chunks').delete().eq('document_id', document_id).execute()
    
    # Process with two-tier chunking
    print("Processing with two-tier chunking...")
    success = await chunker.process_and_save(
        document_id=document_id,
        content=content,
        title=title,
        metadata={
            'source': document.get('source'),
            'source_id': document.get('source_id')
        }
    )
    
    if success:
        print("\n✅ Successfully processed and saved two-tier chunks!")
        
        # Verify the chunks were saved
        chunks_response = supabase.client.table('chunks').select('*').eq('document_id', document_id).execute()
        
        # Group by level
        page_chunks = [c for c in chunks_response.data if c.get('chunk_level') == 'page']
        section_chunks = [c for c in chunks_response.data if c.get('chunk_level') == 'section']
        semantic_chunks = [c for c in chunks_response.data if c.get('chunk_level') == 'semantic']
        
        print(f"\nSaved chunks summary:")
        print(f"  - Page chunks: {len(page_chunks)}")
        print(f"  - Section chunks: {len(section_chunks)}")
        print(f"  - Semantic chunks: {len(semantic_chunks)}")
        print(f"  - Total: {len(chunks_response.data)}")
        
        # Show semantic chunk samples
        if semantic_chunks:
            print("\nSemantic chunk samples:")
            for chunk in semantic_chunks[:3]:
                print(f"\n  - Focus: '{chunk.get('semantic_focus', 'N/A')}'")
                print(f"    Content: {chunk.get('content', '')[:100]}...")
                print(f"    Context: {chunk.get('contextual_summary', '')[:100]}...")
        
        # Update document status
        print("\nUpdating document status to 'chunked'...")
        supabase.client.table('documents').update({
            'status': 'chunked',
            'processing_metadata': {
                'chunking_method': 'two_tier',
                'total_chunks': len(chunks_response.data),
                'page_chunks': len(page_chunks),
                'section_chunks': len(section_chunks),
                'semantic_chunks': len(semantic_chunks)
            }
        }).eq('id', document_id).execute()
        
        print("\n🎉 Document successfully processed with two-tier chunking!")
        print("You can now view the hierarchical chunks in the frontend.")
        
    else:
        print("\n❌ Failed to process and save chunks")


async def process_all_documents():
    """Process all documents with two-tier chunking."""
    
    chunker = TwoTierChunker()
    supabase = SupabaseService()
    
    # Get all documents
    docs_response = supabase.client.table('documents').select('*').execute()
    
    print(f"Found {len(docs_response.data)} documents to process")
    
    for doc in docs_response.data:
        print(f"\nProcessing: {doc.get('title', 'Untitled')}")
        
        try:
            # Clear existing chunks
            supabase.client.table('chunks').delete().eq('document_id', doc['id']).execute()
            
            # Process with two-tier chunking
            success = await chunker.process_and_save(
                document_id=doc['id'],
                content=doc.get('content', ''),
                title=doc.get('title', 'Untitled'),
                metadata={
                    'source': doc.get('source'),
                    'source_id': doc.get('source_id')
                }
            )
            
            if success:
                print(f"  ✅ Successfully processed {doc.get('title')}")
            else:
                print(f"  ❌ Failed to process {doc.get('title')}")
                
        except Exception as e:
            print(f"  ❌ Error processing {doc.get('title')}: {e}")


async def main():
    """Main entry point."""
    print("TWO-TIER CHUNKING PROCESSOR")
    print("="*80)
    
    print("\nOptions:")
    print("1. Process NC2050 document only")
    print("2. Process all documents")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == "1":
        await process_nc2050()
    elif choice == "2":
        await process_all_documents()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
