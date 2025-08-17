"""Setup script for CrediTrust AI Platform."""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "vector_store",
        "models",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {directory}")

def check_data():
    """Check if data file exists."""
    print("🔍 Checking data files...")
    if Path("filtered_data.parquet").exists():
        print("  ✓ filtered_data.parquet found")
        return True
    else:
        print("  ❌ filtered_data.parquet not found")
        print("     Please ensure the data file is in the root directory")
        return False

def main():
    """Main setup function."""
    print("🏦 CrediTrust AI Platform Setup")
    print("=" * 40)
    
    try:
        # Install requirements
        install_requirements()
        
        # Create directories
        create_directories()
        
        # Check data
        data_exists = check_data()
        
        print("\n" + "=" * 40)
        print("✅ Setup completed!")
        
        if data_exists:
            print("\n🚀 Next steps:")
            print("1. Run: python build_vector_store.py")
            print("2. Run: streamlit run app.py")
        else:
            print("\n⚠️  Please add filtered_data.parquet before proceeding")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()