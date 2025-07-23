import subprocess, sys, importlib, pathlib    
import torch, urllib.request, pathlib
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt, seaborn as sns

REPO = "CLIP-dissect"
URL  = "https://github.com/Trustworthy-ML-Lab/CLIP-dissect.git"

if not pathlib.Path(REPO).exists():
    subprocess.run(["git", "clone", "--depth", "1", URL, REPO], check=True)
else:
    print("CLIP‑Dissect repo already present")

PKGS = {
    "torch"                                   : "torch",
    "torchvision"                             : "torchvision",
    "git+https://github.com/openai/CLIP.git"  : "clip",
    "pandas"                                  : "pandas",
    "tqdm"                                    : "tqdm",
    "ftfy"                                    : "ftfy",
    "regex"                                   : "regex",
    "matplotlib"                              : "matplotlib",
    "seaborn"                                 : "seaborn",
    "Pillow"                                  : "PIL",
    "scikit-learn"                            : "sklearn",
    "gdown"                                   : "gdown",
}
for pip_spec, mod_name in PKGS.items():
    try:
        import importlib; importlib.import_module(mod_name)
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_spec], check=True)

repo_path = str(pathlib.Path(REPO).resolve())
if repo_path not in sys.path:
    sys.path.append(repo_path)

print("Environment ready")

import subprocess, pathlib

broden_dir = pathlib.Path("CLIP-dissect/data/broden1_224/images")
if not broden_dir.exists():
    subprocess.run(["bash", "dlbroden.sh"], cwd="CLIP-dissect", check=True)
else:
    print("Broden already present")

# load ResNet-18 pretrained on ImageNet
model_imagenet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()

# ResNet‑18 Places365
places_ckpt = pathlib.Path("CLIP-dissect/data/resnet18_places365.pth.tar")
if not places_ckpt.exists():
    urllib.request.urlretrieve(
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        places_ckpt,
    )

# Define a loader function for Places365
def load_resnet18_places(ckpt):
    m = resnet18(num_classes=365)
    state = torch.load(ckpt, map_location="cpu")["state_dict"]
    m.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return m.eval()

model_places = load_resnet18_places(places_ckpt)
print("Both models loaded")

import subprocess, datetime, glob, os

repo = "CLIP-dissect"
layers = "layer2,layer3,layer4"

def run_dissection(target_name):
    stamp = datetime.datetime.now().strftime("%y_%m_%d")
    out_dir = f"{repo}/results/{target_name}_{stamp}"

  # Avoids re-running dissection if outputs already exist for today’s date and model.
    if glob.glob(f"{out_dir}*"):
        print(f" Cached results → {out_dir}")
        return sorted(glob.glob(f"{out_dir}*"))[-1]

  # Attempts batch sizes 32 then 16 using CUDA.
    for batch in (32, 16):
        cmd = [
            "python", "describe_neurons.py",
            "--clip_model", "ViT-B/16",
            "--target_model", target_name,
            "--target_layers", layers,
            "--d_probe", "broden",
            "--batch_size", str(batch),
            "--device", "cuda",
        ]
        print("Trying:", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=repo)
        if ret.returncode == 0:
            return sorted(glob.glob(f"{repo}/results/{target_name}_{stamp}*"))[-1]

    cmd = [
        "python", "describe_neurons.py",
        "--clip_model", "ViT-B/16",
        "--target_model", target_name,
        "--target_layers", layers,
        "--d_probe", "broden",
        "--batch_size", "8",
        "--device", "cpu",
    ]
    ret = subprocess.run(cmd, cwd=repo)
    if ret.returncode == 0:
        return sorted(glob.glob(f"{repo}/results/{target_name}_{stamp}*"))[-1]
    raise RuntimeError("Dissection failed on all settings")

# Run both models
result_imagenet = run_dissection("resnet18")
result_places   = run_dissection("resnet18_places")

print("Dissection complete:")
print("ImageNet", result_imagenet)
print("Places365", result_places)

import pandas as pd, pathlib

def load_df(folder):
    return pd.read_csv(pathlib.Path(folder) / "descriptions.csv")

df_img = load_df(result_imagenet)
df_pla = load_df(result_places)

def category(tag):
    return {"-o":"object","-p":"part","-s":"scene",
            "-t":"texture","-c":"color"}.get(tag[-2:], "other")

for df in (df_img, df_pla):
    df["category"] = df["description"].apply(category)

print("Top‑5 concepts per model:")

# Displays the most frequent concept labels assigned to neurons.
# Useful to understand what types of concepts dominate in each model.

for name, df in [("ImageNet", df_img), ("Places365", df_pla)]:
    print(f"\n{name}")
    print(df["description"].value_counts().head())

# Finds how many neuron-aligned concepts were shared between both models.
overlap = set(df_img["description"]) & set(df_pla["description"])
print("Shared concepts:", len(overlap))

print(df_pla["category"].value_counts())


plt.figure(figsize=(8,4))
for idx, (title, df) in enumerate([("ImageNet", df_img), ("Places365", df_pla)], 1):
    plt.subplot(1,2,idx)
    sns.countplot(y=df["category"], order=df["category"].value_counts().index)
    plt.title(title); plt.xlabel("# neurons"); plt.tight_layout()
plt.show()

# Finds which concepts are exclusive to each model.
unique_img = set(df_img["description"]) - set(df_pla["description"])
unique_pla = set(df_pla["description"]) - set(df_img["description"])

def plot_unique(unique_set, df, title):
    series = df[df["description"].isin(unique_set)]["description"].value_counts().head(10)
    plt.figure(figsize=(5,3)); series[::-1].plot(kind="barh"); plt.title(title)
    plt.tight_layout(); plt.show()

plot_unique(unique_img, df_img, "Unique to ImageNet")
plot_unique(unique_pla, df_pla, "Unique to Places365")
