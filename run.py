import subprocess
import sys

def main():
    print("🚀 启动 FundingMatch AI...")
    print("=" * 50)
    
    # 检查依赖
    print("📦 检查依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 启动应用
    print("\n🌐 启动 Streamlit 应用...")
    print("=" * 50)
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()