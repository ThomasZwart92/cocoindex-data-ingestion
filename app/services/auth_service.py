"""
Simple auth service placeholder
"""
from typing import Optional

async def get_current_user():
    """Placeholder auth function - returns mock user"""
    return {
        "id": "test-user",
        "email": "test@example.com",
        "security_level": "employee",
        "access_level": 4
    }