import subprocess
import sys

def main():
    print("🚀 start FundingMatch AI...")
    print("=" * 50)
    
    # check dependencies
    print("📦 check dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # start application
    print("\n🌐 start Streamlit application...")
    print("=" * 50)
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()