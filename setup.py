"""
Quick setup script — installs dependencies and pulls the default Ollama model.
"""
import subprocess
import sys

def run(cmd, desc):
    print(f"\n{'='*50}")
    print(f"  {desc}")
    print(f"{'='*50}")
    subprocess.run(cmd, shell=True, check=False)

if __name__ == "__main__":
    # Install Python deps
    run(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python dependencies")
    
    # Pull default Ollama model
    run("ollama pull qwen2.5:3b", "Pulling Ollama model: qwen2.5:3b (≤4B params)")
    
    print("\n" + "="*50)
    print("  Setup complete!")
    print("  Run 'start.bat' or:")
    print("    cd backend && python server.py")
    print("    cd frontend && streamlit run app.py")
    print("="*50)
