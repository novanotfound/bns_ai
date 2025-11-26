"""
SETUP SCRIPT WITH VIRTUAL ENVIRONMENT SUPPORT

NEW LOGIC:
1. Create virtual environment first
2. Install packages inside venv
3. All future commands use venv's Python

WHY THIS FIXES THE ERROR:
- Ubuntu 22.04+ blocks system-wide pip installs
- Virtual environments are isolated and safe
- No risk of breaking system Python
"""

import os
import subprocess
import sys
import platform

def create_virtual_environment():
    """
    Creates an isolated Python environment
    
    WHAT'S HAPPENING:
    1. python3 -m venv venv → Creates a folder called 'venv'
    2. venv contains its own Python interpreter
    3. venv has its own pip (package installer)
    4. Anything installed goes ONLY in venv, not system
    
    FOLDER STRUCTURE AFTER:
    Advocacy_project/
    ├── venv/              ← New isolated environment
    │   ├── bin/           ← Contains python, pip executables
    │   ├── lib/           ← Installed packages go here
    │   └── ...
    └── setup.py
    """
    if os.path.exists("venv"):
        print("✓ Virtual environment already exists")
        return
    
    print("📦 Creating virtual environment...")
    print("   (This creates an isolated Python sandbox)")
    
    # Create venv using system Python
    subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    
    print("✓ Virtual environment created: ./venv")

def get_venv_python():
    """
    Returns the path to the venv's Python executable
    
    WHY DIFFERENT PATHS:
    - Linux/Mac: venv/bin/python
    - Windows: venv\Scripts\python.exe
    
    This function abstracts the OS difference
    """
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "python.exe")
    else:
        return os.path.join("venv", "bin", "python")

def get_venv_pip():
    """Returns path to venv's pip"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "pip.exe")
    else:
        return os.path.join("venv", "bin", "pip")

def create_directory_structure():
    """Creates project folders (same as before)"""
    directories = [
        "data/raw",
        "data/processed", 
        "graphrag_storage",
        "src",
        "logs"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")

def install_dependencies():
    """
    Installs packages using VENV's pip (not system pip)
    
    KEY CHANGE:
    Before: pip install package  (used system pip → ERROR)
    Now:    venv/bin/pip install package  (uses venv pip → SUCCESS)
    """
    packages = [
        "pandas==2.1.4",
        "google-generativeai==0.3.2",
        "nano-graphrag",
        "sentence-transformers==2.2.2",
        "python-dotenv==1.0.0",
        "tqdm==4.66.1",
        "loguru==0.7.2"
    ]
    
    pip_path = get_venv_pip()
    
    print("\n📦 Installing dependencies in virtual environment...")
    print(f"   Using pip: {pip_path}\n")
    
    for package in packages:
        print(f"Installing {package}...")
        # Use venv's pip instead of system pip
        subprocess.check_call([pip_path, "install", package])
    
    print("\n✓ All packages installed in venv")

def setup_env_file():
    """Creates .env file (same as before)"""
    env_content = """# Gemini API Configuration
GOOGLE_API_KEY=your_api_key_here

# GraphRAG Configuration  
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("\n✓ Created .env file")
    print("⚠️  ACTION REQUIRED: Edit .env and add your Gemini API key")

def validate_csv():
    """
    Validates BNS CSV (now using venv's Python)
    
    CRITICAL: We must import pandas from VENV, not system
    That's why we use subprocess to call venv's Python
    """
    csv_path = "bns_sections.csv"
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: {csv_path} not found")
        print("   Place your CSV in the project root")
        return False
    
    # Use venv's Python to import pandas
    validation_script = """
import pandas as pd
import sys

df = pd.read_csv('bns_sections.csv')

required_columns = ["Chapter", "Chapter_name", "Section", "Section_name", "Description"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    print(f"ERROR: Missing columns: {missing}")
    sys.exit(1)

print(f"SUCCESS: {len(df)} sections found")
print(f"Chapters: {df['Chapter'].nunique()}")
print(f"Sample: {df['Section'].iloc[0]} - {df['Section_name'].iloc[0]}")
"""
    
    venv_python = get_venv_python()
    
    try:
        result = subprocess.run(
            [venv_python, "-c", validation_script],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"\n✓ CSV validated:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ CSV validation failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def create_activation_helper():
    """
    Creates a helper script to activate venv easily
    
    WHY THIS HELPS:
    Instead of typing: source venv/bin/activate
    User can type: source activate.sh
    """
    if platform.system() == "Windows":
        content = "@echo off\ncall venv\\Scripts\\activate.bat"
        filename = "activate.bat"
    else:
        content = "#!/bin/bash\nsource venv/bin/activate"
        filename = "activate.sh"
    
    with open(filename, "w") as f:
        f.write(content)
    
    if platform.system() != "Windows":
        os.chmod(filename, 0o755)  # Make executable
    
    print(f"✓ Created activation helper: {filename}")

if __name__ == "__main__":
    print("🚀 BNS Legal Expert - Project Setup\n")
    print("="*50)
    
    # Step 1: Create venv FIRST
    create_virtual_environment()
    
    # Step 2: Create folders
    create_directory_structure()
    
    # Step 3: Install packages in venv
    install_dependencies()
    
    # Step 4: Setup config
    setup_env_file()
    
    # Step 5: Create helper script
    create_activation_helper()
    
    print("\n" + "="*50)
    print("\n📋 NEXT STEPS:")
    print("1. Edit .env file and add your Gemini API key")
    print("2. Activate virtual environment:")
    
    if platform.system() == "Windows":
        print("   activate.bat")
    else:
        print("   source activate.sh")
    
    print("3. Run: python src/data_preparation.py")
    print("\n" + "="*50)
    
    # Validate CSV
    validate_csv()