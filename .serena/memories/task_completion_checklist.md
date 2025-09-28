# Task Completion Checklist

When completing any development task, follow this checklist to ensure quality and maintainability:

## 1. Code Quality Checks

### Before Committing
- [ ] Run formatter: `black app/ tests/` (Python)
- [ ] Run linter: `ruff check app/ tests/`
- [ ] Type checking: `mypy app/` (if configured)
- [ ] Remove debug print statements
- [ ] Remove commented-out code
- [ ] Ensure no hardcoded credentials/secrets

### Frontend Checks
- [ ] Run formatter: `npm run format` (if configured)
- [ ] ESLint: `npm run lint`
- [ ] Build check: `npm run build` (should complete without errors)

## 2. Testing Requirements

### Unit Tests
- [ ] Write tests for new functions/methods
- [ ] Ensure existing tests still pass: `pytest`
- [ ] Aim for >80% code coverage on new code
- [ ] Test edge cases and error conditions

### Integration Tests
- [ ] Test with real database connections
- [ ] Test API endpoints with various inputs
- [ ] Verify state transitions work correctly

### Manual Testing
- [ ] Test the feature end-to-end in browser
- [ ] Check console for JavaScript errors
- [ ] Verify UI updates correctly
- [ ] Test error states and loading states

## 3. Documentation Updates

### Code Documentation
- [ ] Add/update docstrings for new functions
- [ ] Update type hints
- [ ] Add inline comments for complex logic

### Project Documentation
- [ ] Update README.md if needed
- [ ] Update architecture.md for structural changes
- [ ] Document new environment variables in .env.example
- [ ] Update API documentation for new endpoints

## 4. Database Considerations

### Schema Changes
- [ ] Create migration scripts for schema changes
- [ ] Test migrations on fresh database
- [ ] Update supabase_schema.sql if needed
- [ ] Verify indexes for new queries

### Data Integrity
- [ ] Ensure foreign key constraints are proper
- [ ] Add database constraints where needed
- [ ] Consider adding database triggers for audit

## 5. Performance Checks

### Backend Performance
- [ ] Profile slow queries (use EXPLAIN ANALYZE)
- [ ] Check for N+1 query problems
- [ ] Ensure proper use of async/await
- [ ] Monitor memory usage for large datasets

### Frontend Performance
- [ ] Check bundle size impact
- [ ] Lazy load heavy components
- [ ] Optimize images and assets
- [ ] Check for unnecessary re-renders

## 6. Security Review

### API Security
- [ ] Validate all input data
- [ ] Check authentication/authorization
- [ ] Prevent SQL injection
- [ ] Rate limiting on expensive operations

### Frontend Security
- [ ] Sanitize user inputs
- [ ] Use HTTPS for all external calls
- [ ] No sensitive data in localStorage
- [ ] CSP headers configured properly

## 7. Deployment Preparation

### Environment Setup
- [ ] Update .env.example with new variables
- [ ] Document any new dependencies
- [ ] Update requirements.txt or package.json
- [ ] Test with production-like settings

### Docker Considerations
- [ ] Update Dockerfile if needed
- [ ] Test docker-compose setup
- [ ] Verify health checks work
- [ ] Check resource limits are appropriate

## 8. Final Checklist

### Before Creating PR
- [ ] Self-review your code changes
- [ ] Run full test suite: `pytest`
- [ ] Check for merge conflicts
- [ ] Write clear PR description
- [ ] Link related issues

### After Deployment
- [ ] Monitor error logs
- [ ] Check performance metrics
- [ ] Verify feature works in production
- [ ] Update status in project tracker

## Quick Validation Commands

```bash
# Full validation suite
pytest                           # Run all tests
black app/ tests/ --check       # Check formatting
ruff check app/ tests/          # Lint code
python app/config_validator.py  # Validate config

# Frontend validation
cd frontend
npm run lint                    # Lint frontend
npm run build                   # Build check
```

## Important Notes

1. **Never skip tests** - They catch issues early
2. **Document breaking changes** - Help team members adapt
3. **Consider rollback plan** - How to revert if issues arise
4. **Communicate blockers** - Ask for help when stuck
5. **Incremental commits** - Small, focused changes are easier to review