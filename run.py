"""Quick run script for CrediTrust AI Platform."""

import subprocess
import sys
from pathlib import Path

def check_setup():
    """Check if system is properly set up."""
    checks = {
        "Data file": Path("filtered_data.parquet").exists(),
        "Vector store": Path("vector_store/faiss_index").exists(),
        "Config file": Path("config/config.yaml").exists()
    }
    
    print("System Status Check:")
    all_good = True
    for check, status in checks.items():
        status_icon = "Correct" if status else "Incorrect"
        print(f"  {status_icon} {check}")
        if not status:
            all_good = False
    
    return all_good

def main():
    """Main run function."""
    print("🏦 CrediTrust AI Platform")
    print("=" * 30)
    
    if not check_setup():
        print("\n⚠️  Setup incomplete. Please run:")
        print("1. python setup.py")
        print("2. python build_vector_store.py")
        return
    
    print("\n🚀 Starting application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()