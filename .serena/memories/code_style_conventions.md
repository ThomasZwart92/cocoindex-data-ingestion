# Code Style and Conventions

## Python Code Style

### General Guidelines
- **Style Guide**: PEP 8 compliance
- **Line Length**: Maximum 120 characters
- **Imports**: Organized with isort (standard library, third-party, local)
- **Type Hints**: Required for all function signatures
- **Docstrings**: Google style docstrings for all public functions/classes

### Naming Conventions
- **Files**: lowercase with underscores (e.g., `document_processor.py`)
- **Classes**: PascalCase (e.g., `DocumentProcessor`)
- **Functions**: lowercase with underscores (e.g., `process_document`)
- **Constants**: UPPERCASE with underscores (e.g., `MAX_CHUNK_SIZE`)
- **Private methods**: Leading underscore (e.g., `_internal_method`)

### CocoIndex-Specific Patterns
```python
# ALWAYS use declarative dataflow pattern
@cocoindex.flow_def(name="FlowName")
def flow_name(flow_builder: FlowBuilder, data_scope: DataScope):
    # Source definition
    data_scope["documents"] = flow_builder.add_source(...)
    
    # Transformation with row context
    with data_scope["documents"].row() as doc:
        doc["field"] = doc["source"].transform(function)
    
    # Export to targets
    collector.export("name", target)
```

### Async Patterns
```python
# Use async/await for I/O operations
async def fetch_document(doc_id: str) -> Document:
    async with aiohttp.ClientSession() as session:
        # Implementation
        pass
```

### Error Handling
```python
# Use specific exceptions with context
try:
    result = await process_document(doc)
except DocumentParsingError as e:
    logger.error(f"Failed to parse document {doc.id}: {e}")
    raise
```

## Frontend Code Style (TypeScript/React)

### General Guidelines
- **Framework**: Next.js 14+ with App Router
- **Components**: Functional components with hooks
- **Styling**: Tailwind CSS utilities
- **State Management**: React hooks (useState, useEffect)

### File Structure
```
frontend/
├── app/           # Next.js app directory
├── components/    # Reusable components
├── lib/          # Utilities and helpers
└── public/       # Static assets
```

### Component Patterns
```typescript
// Use functional components
export default function ComponentName({ prop1, prop2 }: Props) {
  const [state, setState] = useState<Type>(initial);
  
  useEffect(() => {
    // Side effects
  }, [dependencies]);
  
  return (
    <div className="tailwind-classes">
      {/* Content */}
    </div>
  );
}
```

## Database Conventions

### Table Naming
- Plural names: `documents`, `chunks`, `entities`
- Join tables: `document_chunks`, `entity_relationships`

### Column Naming
- lowercase with underscores: `created_at`, `document_id`
- Foreign keys: `{table}_id` (e.g., `document_id`)
- Timestamps: `created_at`, `updated_at`

## Git Conventions

### Branch Naming
- Feature: `feature/description`
- Bug fix: `fix/description`
- Hotfix: `hotfix/description`

### Commit Messages
- Format: `type: description`
- Types: feat, fix, docs, style, refactor, test, chore
- Example: `feat: add three-tier chunking implementation`

## Testing Conventions

### Test File Naming
- Pattern: `test_{module_name}.py`
- Location: `/tests` directory

### Test Structure
```python
def test_function_name_scenario():
    # Arrange
    setup_data = create_test_data()
    
    # Act
    result = function_under_test(setup_data)
    
    # Assert
    assert result.status == expected_status
```

### Test Categories
- Unit tests: Test single functions/methods
- Integration tests: Test component interactions
- E2E tests: Test full workflows