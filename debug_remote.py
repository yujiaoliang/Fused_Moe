import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("fused-moe-debug")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    .add_local_dir("solution", remote_path="/root/solution")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

@app.function(image=image, gpu="B200:1", timeout=600)
def debug_run():
    import sys
    sys.path.insert(0, "/root")
    # Actually, we can just run the function we created!
    with open("/root/debug_tensors.py", "w") as f:
        f.write(Path("debug_tensors.py").read_text())
    
    import subprocess
    result = subprocess.run([sys.executable, "/root/debug_tensors.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return "ERROR: " + result.stderr
    return result.stdout

@app.local_entrypoint()
def main():
    print("\nRunning trace extraction on Modal B200...")
    result = debug_run.remote()
    print("\n=== DEBUG RESULTS ===")
    print(result)

if __name__ == "__main__":
    with app.run():
        main()
