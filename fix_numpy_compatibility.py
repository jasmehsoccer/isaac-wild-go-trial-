#!/usr/bin/env python3
"""
Fix NumPy compatibility issue in IsaacGym torch_utils.py
This script replaces deprecated np.float with float to fix the AttributeError.
"""

import os
import re

def fix_torch_utils():
    """Fix the NumPy compatibility issue in IsaacGym torch_utils.py"""
    
    # Common paths where IsaacGym might be installed
    possible_paths = [
        "/workspace/isaacgym/python/isaacgym/torch_utils.py",
        "/usr/local/lib/python3.8/site-packages/isaacgym/torch_utils.py",
        "/opt/conda/envs/isaac_py38/lib/python3.8/site-packages/isaacgym/torch_utils.py",
        "/root/miniconda3/envs/isaac_py38/lib/python3.8/site-packages/isaacgym/torch_utils.py",
        "isaacgym/python/isaacgym/torch_utils.py"
    ]
    
    torch_utils_path = None
    for path in possible_paths:
        if os.path.exists(path):
            torch_utils_path = path
            break
    
    if not torch_utils_path:
        print("❌ Could not find isaacgym/torch_utils.py")
        print("Please run this script from the directory containing IsaacGym")
        return False
    
    print(f"✅ Found torch_utils.py at: {torch_utils_path}")
    
    # Read the file
    try:
        with open(torch_utils_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Create backup
    backup_path = torch_utils_path + '.backup'
    try:
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"✅ Created backup at: {backup_path}")
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return False
    
    # Fix the deprecated np.float usage
    # Replace np.float with float in function signatures
    fixed_content = re.sub(r'np\.float', 'float', content)
    
    # Check if any changes were made
    if fixed_content == content:
        print("ℹ️  No np.float found to replace")
        return True
    
    # Write the fixed content
    try:
        with open(torch_utils_path, 'w') as f:
            f.write(fixed_content)
        print("✅ Fixed torch_utils.py - replaced np.float with float")
        return True
    except Exception as e:
        print(f"❌ Error writing fixed file: {e}")
        # Restore from backup
        try:
            with open(backup_path, 'r') as f:
                backup_content = f.read()
            with open(torch_utils_path, 'w') as f:
                f.write(backup_content)
            print("✅ Restored original file from backup")
        except Exception as restore_e:
            print(f"❌ Error restoring from backup: {restore_e}")
        return False

def main():
    print("🔧 Fixing NumPy compatibility issue in IsaacGym...")
    
    if fix_torch_utils():
        print("\n✅ Successfully fixed NumPy compatibility issue!")
        print("You can now run the training script.")
    else:
        print("\n❌ Failed to fix the issue.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 