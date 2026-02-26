#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Key Manager with Encryption
Handles secure storage and retrieval of API keys
"""

import os
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages encrypted API keys stored in .env file"""
    
    def __init__(self, env_path: str = None):
        self.env_path = Path(env_path or ".env")
        self.salt_path = Path(".env.salt")
        self._fernet = None
        
    def _get_encryption_key(self) -> bytes:
        """Generate or retrieve encryption key"""
        # Use machine-specific salt
        if not self.salt_path.exists():
            salt = os.urandom(16)
            self.salt_path.write_bytes(salt)
        else:
            salt = self.salt_path.read_bytes()
        
        # Derive key from machine ID + salt
        try:
            machine_id = self._get_machine_id()
        except:
            machine_id = "default_machine_id"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_id.encode()))
        return key
    
    def _get_machine_id(self) -> str:
        """Get unique machine identifier"""
        import uuid
        import platform
        
        # Try to get a stable machine ID
        if platform.system() == "Windows":
            try:
                import subprocess
                result = subprocess.check_output("wmic csproduct get uuid", shell=True)
                return result.decode().split("\n")[1].strip()
            except:
                pass
        elif platform.system() == "Darwin":  # macOS
            try:
                import subprocess
                result = subprocess.check_output("ioreg -rd1 -c IOPlatformExpertDevice | grep IOPlatformUUID", shell=True)
                return result.decode().split('"')[3]
            except:
                pass
        elif platform.system() == "Linux":
            try:
                with open("/etc/machine-id", "r") as f:
                    return f.read().strip()
            except:
                pass
        
        # Fallback to MAC address
        return str(uuid.getnode())
    
    @property
    def fernet(self) -> Fernet:
        """Lazy load Fernet instance"""
        if self._fernet is None:
            key = self._get_encryption_key()
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a string value"""
        if not value:
            return ""
        encrypted = self.fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a string value"""
        if not encrypted_value:
            return ""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return ""
    
    def save_api_key(self, key_name: str, api_key: str):
        """Save encrypted API key to .env file"""
        encrypted = self.encrypt_value(api_key)
        
        # Read existing .env content
        env_content = {}
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        env_content[k.strip()] = v.strip()
        
        # Update key
        env_content[key_name] = encrypted
        
        # Write back
        with open(self.env_path, 'w') as f:
            f.write("# Encrypted API Keys - DO NOT EDIT MANUALLY\n")
            f.write("# Use the Settings tab in the application to manage keys\n\n")
            for k, v in env_content.items():
                f.write(f"{k}={v}\n")
        
        logger.info(f"API key '{key_name}' saved successfully (encrypted)")
    
    def get_api_key(self, key_name: str) -> str:
        """Retrieve and decrypt API key from .env file"""
        if not self.env_path.exists():
            return ""
        
        with open(self.env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip() == key_name:
                        return self.decrypt_value(v.strip())
        
        return ""
    
    def delete_api_key(self, key_name: str):
        """Remove API key from .env file"""
        if not self.env_path.exists():
            return
        
        env_content = {}
        with open(self.env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip() != key_name:
                        env_content[k.strip()] = v.strip()
        
        with open(self.env_path, 'w') as f:
            f.write("# Encrypted API Keys - DO NOT EDIT MANUALLY\n\n")
            for k, v in env_content.items():
                f.write(f"{k}={v}\n")
        
        logger.info(f"API key '{key_name}' deleted")


# Global instance
api_key_manager = APIKeyManager()
