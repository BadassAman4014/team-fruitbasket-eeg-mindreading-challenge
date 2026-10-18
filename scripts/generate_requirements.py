import sys
import subprocess
import importlib.metadata

# Run pip list
proc = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True, encoding="utf-8")
lines = proc.stdout.splitlines()

clean_reqs = []
for line in lines:
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if "==" in line:
        pkg, ver = line.split("==", 1)
        clean_reqs.append((pkg, ver))
    elif "@" in line:
        pkg = line.split("@")[0].strip()
        try:
            ver = importlib.metadata.version(pkg)
            clean_reqs.append((pkg, ver))
        except Exception:
            clean_reqs.append((pkg, ""))
    else:
        clean_reqs.append((line, ""))

# Sort alphabetically by package name (case-insensitive)
clean_reqs.sort(key=lambda x: x[0].lower())

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("# Environment: fruitbasket\n")
    f.write("# Compatible with Python 3.11 / 3.10\n")
    f.write("# PyTorch with CUDA 12.4 enabled\n\n")
    for pkg, ver in clean_reqs:
        if ver:
            f.write(f"{pkg}=={ver}\n")
        else:
            f.write(f"{pkg}\n")

print(f"Total packages written: {len(clean_reqs)}")
for key in ["torch", "torchvision", "torchaudio", "numpy", "scipy", "scikit-learn", "tqdm", "einops"]:
    found = [f"{p}=={v}" for p, v in clean_reqs if p.lower() == key.lower()]
    print(f"  {key}: {found}")
