import subprocess
result = subprocess.run(["C:\\Users\\jiaya\\anaconda3\\envs\\fi-bench\\python.exe", "-X", "utf8", "-m", "modal", "run", "scripts/profile_modal.py"], cwd="d:\\Research\\mlsys_note", capture_output=True, text=True, encoding="utf-8", errors="replace")
with open("profile_output.txt", "w", encoding="utf-8") as f:
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
print("Done profiling")
