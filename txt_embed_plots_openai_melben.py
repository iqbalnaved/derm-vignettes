import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from openai import OpenAI
import numpy as np

# Initialize OpenAI client (ensure your API key is set)
openai_api_key = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=openai_api_key)

# --- Load JSON files and label by source ---
def load_vignettes(file_path, source_label):
    with open(file_path, 'r') as f:
        data = json.load(f)
    vignettes, labels, sources = [], [], []
    for item in data:
        diag = item["diagnosis"].strip().lower()
        if diag in ["melanoma", "nevus"]:
            vignettes.append(item["vignette"])
            labels.append(diag.capitalize())
            sources.append(source_label)
    return vignettes, labels, sources

# Load datasets
v_img, l_img, s_img = load_vignettes("/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_1_10132025.json", "With image")
v_txt, l_txt, s_txt = load_vignettes("/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_3_10312025.json", "Without image")

vignettes = v_img + v_txt
labels = l_img + l_txt
sources = s_img + s_txt

print(f"Loaded {len(vignettes)} total vignettes.")

# --- Get embeddings using OpenAI's text-embedding-3-small ---
def get_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings = [d.embedding for d in response.data]
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

print("Generating embeddings with text-embedding-3-small...")
embeddings = get_embeddings(vignettes)
print("Embeddings shape:", embeddings.shape)

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

# --- Combine metadata ---
df = pd.DataFrame({
    "Diagnosis": labels,
    "Source": sources,
    "TSNE1": tsne_result[:, 0],
    "TSNE2": tsne_result[:, 1]
})
df["Group"] = df["Source"] + " – " + df["Diagnosis"]

# --- Color map ---
color_map = {
    "With image": "#E74C3C",     # red
    "Without image": "#3498DB"   # blue
}

# --- Timestamp for filenames ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

# --- Plot helper for a single diagnosis ---
def plot_diagnosis(df, diagnosis, filename):
    plt.figure(figsize=(8, 6))
    subset = df[df["Diagnosis"] == diagnosis]
    for source, color in color_map.items():
        sdata = subset[subset["Source"] == source]
        plt.scatter(
            sdata["TSNE1"], sdata["TSNE2"],
            c=color, label=source, s=60, alpha=0.8, edgecolors='k'
        )
    plt.title(f"{diagnosis} – With Image vs Without Image (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Source", loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# --- Create and save two plots ---
plot_diagnosis(df, "Melanoma", f"/mnt/d/Naved/Outputs/derm_vignettes/tsne_melanoma_{timestamp}.png")
plot_diagnosis(df, "Nevus", f"/mnt/d/Naved/Outputs/derm_vignettes/tsne_nevus_{timestamp}.png")
