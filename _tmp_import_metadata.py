import importlib
import sys
try:
    import app.services.metadata_extraction_service as mes
    print('service import ok')
    import app.flows.metadata_extraction_flow as mef
    print('flow import ok')
except Exception as e:
    print('import error:', e)
    sys.exit(1)
