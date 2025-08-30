"""Path validation utilities for secure file operations"""
import os
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

class PathSecurityError(Exception):
    """Raised when a path operation is deemed insecure"""
    pass

class PathValidator:
    """Validate and sanitize file paths for secure operations"""
    
    @staticmethod
    def validate_path(
        path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None,
        allow_urls: bool = True
    ) -> str:
        """
        Validate and sanitize a file path to prevent directory traversal attacks.
        
        Args:
            path: The path to validate
            base_dir: Optional base directory to restrict access to
            allow_urls: Whether to allow URL paths (http/https)
            
        Returns:
            Sanitized path as string
            
        Raises:
            PathSecurityError: If path is deemed insecure
        """
        if not path:
            raise PathSecurityError("Empty path provided")
        
        path_str = str(path)
        
        # Check if it's a URL
        if path_str.startswith(('http://', 'https://')):
            if not allow_urls:
                raise PathSecurityError("URL paths are not allowed")
            
            # Validate URL
            try:
                parsed = urlparse(path_str)
                if not parsed.scheme or not parsed.netloc:
                    raise PathSecurityError(f"Invalid URL: {path_str}")
                return path_str
            except Exception as e:
                raise PathSecurityError(f"Invalid URL: {e}")
        
        # Handle local file paths
        try:
            # Resolve to absolute path and remove any .. or symbolic links
            resolved_path = Path(path_str).resolve()
            
            # If base_dir is specified, ensure path is within it
            if base_dir:
                base = Path(base_dir).resolve()
                
                # Check if resolved path is within base directory
                try:
                    resolved_path.relative_to(base)
                except ValueError:
                    raise PathSecurityError(
                        f"Path '{path_str}' is outside allowed directory '{base_dir}'"
                    )
            
            # Check for suspicious patterns
            path_str_lower = str(resolved_path).lower()
            suspicious_patterns = [
                '/etc/passwd',
                '/etc/shadow',
                'C:\\Windows\\System32',
                'C:\\Windows\\System',
                '/proc/',
                '/sys/',
                '\\..\\',
                '/../',
                '\x00',  # Null bytes
            ]
            
            for pattern in suspicious_patterns:
                if pattern.lower() in path_str_lower:
                    raise PathSecurityError(
                        f"Path contains suspicious pattern: {pattern}"
                    )
            
            # Also validate the filename component
            filename = os.path.basename(str(resolved_path))
            if filename:
                PathValidator.validate_filename(filename)
            
            return str(resolved_path)
            
        except PathSecurityError:
            raise
        except Exception as e:
            raise PathSecurityError(f"Invalid path: {e}")
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """
        Validate and sanitize a filename to prevent security issues.
        
        Args:
            filename: The filename to validate
            
        Returns:
            Sanitized filename
            
        Raises:
            PathSecurityError: If filename is deemed insecure
        """
        if not filename:
            raise PathSecurityError("Empty filename provided")
        
        # Remove any directory components
        filename = os.path.basename(filename)
        
        # Check for suspicious patterns
        forbidden_chars = ['/', '\\', '\x00', ':', '*', '?', '"', '<', '>', '|']
        for char in forbidden_chars:
            if char in filename:
                raise PathSecurityError(
                    f"Filename contains forbidden character: {repr(char)}"
                )
        
        # Check for reserved Windows filenames
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        name_without_ext = filename.split('.')[0].upper()
        if name_without_ext in reserved_names:
            raise PathSecurityError(f"Filename is a reserved name: {filename}")
        
        # Limit filename length
        if len(filename) > 255:
            raise PathSecurityError(f"Filename too long: {len(filename)} characters")
        
        # Don't allow hidden files in Unix-like systems
        if filename.startswith('.'):
            raise PathSecurityError("Hidden files are not allowed")
        
        return filename
    
    @staticmethod
    def ensure_safe_directory(directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists and is safe to use.
        
        Args:
            directory: Directory path to validate and create if needed
            
        Returns:
            Path object for the validated directory
            
        Raises:
            PathSecurityError: If directory is unsafe
        """
        validated_path = PathValidator.validate_path(directory, allow_urls=False)
        dir_path = Path(validated_path)
        
        # Create directory if it doesn't exist
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise PathSecurityError(f"Failed to create directory: {e}")
        
        # Verify it's actually a directory
        if not dir_path.is_dir():
            raise PathSecurityError(f"Path exists but is not a directory: {directory}")
        
        # Check permissions (basic check)
        if not os.access(dir_path, os.W_OK | os.R_OK):
            raise PathSecurityError(f"Insufficient permissions for directory: {directory}")
        
        return dir_path