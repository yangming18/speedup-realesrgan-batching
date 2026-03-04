#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dependency Compatibility Patch
Automatically resolves version conflicts between packages
"""

import subprocess
import sys
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# Define compatible versions for conflicting packages
COMPATIBLE_VERSIONS = {
    'numpy': '>=1.24.4,<2',
    'Pillow': '>=10.0.0,<12.0',
    'opencv-python': '==4.9.0.80',
    'opencv-python-headless': '>=4.9.0.80',
    'rembg': '==2.0.72'
}


def check_package_version(package_name: str) -> Tuple[bool, str]:
    """
    Check if a package is installed and get its version.
    
    Returns:
        Tuple of (installed, version)
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    return True, version
        
        return False, ''
    except Exception as e:
        logger.warning(f"Could not check {package_name}: {e}")
        return False, ''


def fix_dependencies(verbose: bool = False) -> bool:
    """
    Fix dependency conflicts automatically.
    
    Args:
        verbose: Print detailed information
    
    Returns:
        True if any fixes were applied
    """
    fixes_applied = False
    
    if verbose:
        print("🔍 Checking dependencies for conflicts...")
    
    # Check if rembg is installed (trigger for fixing)
    rembg_installed, _ = check_package_version('rembg')
    
    if not rembg_installed:
        # No rembg, no conflicts expected
        if verbose:
            print("✓ No rembg detected, dependencies should be fine")
        return False
    
    if verbose:
        print("📦 rembg detected, checking for conflicts...")
    
    # Check numpy version
    numpy_installed, numpy_version = check_package_version('numpy')
    if numpy_installed and numpy_version.startswith('2.'):
        if verbose:
            print(f"⚠️  numpy {numpy_version} is incompatible, fixing...")
        _fix_package('numpy', COMPATIBLE_VERSIONS['numpy'])
        fixes_applied = True
    
    # Check Pillow version
    pillow_installed, pillow_version = check_package_version('Pillow')
    if pillow_installed:
        major_version = int(pillow_version.split('.')[0])
        if major_version >= 12:
            if verbose:
                print(f"⚠️  Pillow {pillow_version} is incompatible, fixing...")
            _fix_package('Pillow', COMPATIBLE_VERSIONS['Pillow'])
            fixes_applied = True
    
    # Check opencv versions
    opencv_installed, opencv_version = check_package_version('opencv-python')
    if opencv_installed and opencv_version != '4.9.0.80':
        if verbose:
            print(f"⚠️  opencv-python {opencv_version} is incompatible, fixing...")
        _fix_package('opencv-python', COMPATIBLE_VERSIONS['opencv-python'])
        fixes_applied = True
    
    # Install opencv-python-headless if missing (required by albumentations)
    headless_installed, _ = check_package_version('opencv-python-headless')
    if not headless_installed:
        if verbose:
            print("📦 Installing opencv-python-headless...")
        _fix_package('opencv-python-headless', COMPATIBLE_VERSIONS['opencv-python-headless'])
        fixes_applied = True
    
    if fixes_applied:
        if verbose:
            print("✅ Dependencies fixed!")
    else:
        if verbose:
            print("✓ All dependencies are compatible")
    
    return fixes_applied


def _fix_package(package_name: str, version_spec: str):
    """Install or fix a specific package version"""
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--force-reinstall', 
             '--no-deps', f'{package_name}{version_spec}'],
            capture_output=True,
            timeout=120
        )
        # Reinstall dependencies if needed
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', f'{package_name}{version_spec}'],
            capture_output=True,
            timeout=120
        )
    except Exception as e:
        logger.error(f"Failed to fix {package_name}: {e}")


def apply_dependency_patch(silent: bool = True):
    """
    Apply dependency compatibility patch.
    Called automatically at application startup.
    
    Args:
        silent: If True, only log warnings/errors. If False, print status.
    """
    try:
        fixes_applied = fix_dependencies(verbose=not silent)
        
        if fixes_applied:
            logger.info("Dependency compatibility patch applied successfully")
        else:
            logger.debug("No dependency fixes needed")
            
    except Exception as e:
        logger.warning(f"Could not apply dependency patch: {e}")


if __name__ == "__main__":
    # Run as standalone script
    print("=" * 60)
    print("Dependency Compatibility Patch")
    print("=" * 60)
    print()
    
    fix_dependencies(verbose=True)
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
