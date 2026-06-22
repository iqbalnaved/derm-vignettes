import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from openai import OpenAI
import numpy as np

# Initialize OpenAI client
openai_api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=openai_api_key)

# --- Load JSON files and label by source ---
def load_vignettes(file_path, source_label):
    with open(file_path, 'r') as f:
        data = json.load(f)
    vignettes, labels, sources = [], [], []
    valid_diseases = [
        "melanoma",
        "nevus",
        "basal cell carcinoma",
        "dermatofibroma",
        "seborrheic keratosis",
        "squamous cell carcinoma"
    ]
    for item in data:
        diag = item["diagnosis"].strip().lower()
        if diag in valid_diseases:
            vignettes.append(item["vignette"])
            labels.append(diag.title())
            sources.append(source_label)
    return vignettes, labels, sources

# Load datasets
v_img, l_img, s_img = load_vignettes(
    "/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_1_10132025.json", "With image"
)
v_txt, l_txt, s_txt = load_vignettes(
    "/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_3_11042025_v1.json", "Without image"
)

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

# --- Combine metadata ---
df = pd.DataFrame({
    "Diagnosis": labels,
    "Source": sources,
})
df["Embedding"] = list(embeddings)

# --- Marker mapping (filled for with image, unfilled for without) ---
shape_map = {
    "Melanoma": "^",              # upward triangle
    "Nevus": "v",                 # downward triangle
    "Basal Cell Carcinoma": "*",  # star
    "Dermatofibroma": "o",        # circle
    "Seborrheic Keratosis": "s",  # square
    "Squamous Cell Carcinoma": "D" # diamond
}

# --- t-SNE helper ---
def run_tsne(embeddings, perplexity=30):
    n_samples = embeddings.shape[0]
    if n_samples <= perplexity:
        perplexity = max(5, n_samples // 3)
        print(f"Adjusted perplexity to {perplexity} for {n_samples} samples")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(embeddings)

# --- Plot combined t-SNE for all diseases ---
def plot_all_diseases(df, filename):
    tsne_result = run_tsne(np.vstack(df["Embedding"].values))
    df_plot = df.copy()
    df_plot["TSNE1"] = tsne_result[:, 0]
    df_plot["TSNE2"] = tsne_result[:, 1]

    plt.figure(figsize=(10, 8))

    for diag, marker in shape_map.items():
        for source in ["With image", "Without image"]:
            subset = df_plot[(df_plot["Diagnosis"] == diag) & (df_plot["Source"] == source)]
            if len(subset) == 0:
                continue

            facecolor = 'k' if source == "With image" else 'none'
            edgecolor = 'k'

            plt.scatter(
                subset["TSNE1"], subset["TSNE2"],
                marker=marker, s=80,
                facecolors=facecolor, edgecolors=edgecolor,
                alpha=0.8, label=f"{diag} ({source})"
            )

    plt.title("t-SNE Visualization – 6 Skin Conditions (With/Without Image)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# --- Plot per-disease comparison (with vs without image) ---
def plot_disease(df, diagnosis, filename):
    subset = df[df["Diagnosis"] == diagnosis]
    if len(subset) < 2:
        print(f"Skipping {diagnosis}: not enough data.")
        return

    tsne_result = run_tsne(np.vstack(subset["Embedding"].values))
    subset = subset.copy()
    subset["TSNE1"] = tsne_result[:, 0]
    subset["TSNE2"] = tsne_result[:, 1]

    plt.figure(figsize=(8, 6))

    for source in ["With image", "Without image"]:
        sdata = subset[subset["Source"] == source]
        if len(sdata) == 0:
            continue

        marker = shape_map[diagnosis]
        facecolor = 'k' if source == "With image" else 'none'
        edgecolor = 'k'

        plt.scatter(
            sdata["TSNE1"], sdata["TSNE2"],
            marker=marker, s=100,
            facecolors=facecolor, edgecolors=edgecolor,
            alpha=0.85, label=source
        )

    plt.title(f"{diagnosis} – With vs Without Image (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Source", loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# --- Run plots ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
output_dir = "/mnt/d/Naved/Outputs/derm_vignettes"

# Combined plot
plot_all_diseases(df, f"{output_dir}/tsne_all_6_conditions_{timestamp}.png")

# Individual plots per diagnosis
for diag in shape_map.keys():
    plot_disease(df, diag, f"{output_dir}/tsne_{diag.replace(' ', '_').lower()}_{timestamp}.png")
