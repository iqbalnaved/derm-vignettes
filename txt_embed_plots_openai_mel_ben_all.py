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
v_img, l_img, s_img = load_vignettes(
    "/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_1_10132025.json", "With image"
)
v_txt, l_txt, s_txt = load_vignettes(
    "/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_3_11032025_v2.json", "Without image"
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
df["Group"] = df["Source"] + " – " + df["Diagnosis"]
df["Embedding"] = list(embeddings)

# --- Color maps ---
pair_color_map = {
    "With image": "#E74C3C",     # red
    "Without image": "#3498DB"   # blue
}

combined_color_map = {
    "With image – Melanoma": "#E74C3C",     # red
    "With image – Nevus": "#3498DB",        # blue
    "Without image – Melanoma": "#2ECC71",  # green
    "Without image – Nevus": "#F1C40F"      # yellow
}

# --- Timestamp for filenames ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
output_dir = "/mnt/d/Naved/Outputs/derm_vignettes"

# --- t-SNE helper ---
def run_tsne(embeddings, perplexity=10):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(embeddings)

# --- Plot helper for single diagnosis (2-color contrast) ---
def plot_diagnosis(df, diagnosis, filename):
    subset = df[df["Diagnosis"] == diagnosis]
    tsne_result = run_tsne(np.vstack(subset["Embedding"].values))

    subset = subset.copy()
    subset["TSNE1"] = tsne_result[:, 0]
    subset["TSNE2"] = tsne_result[:, 1]

    plt.figure(figsize=(8, 6))
    for source, color in pair_color_map.items():
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

# --- Plot helper for combined plot (4 categories) ---
def plot_combined(df, filename):
    tsne_result = run_tsne(np.vstack(df["Embedding"].values))
    df_plot = df.copy()
    df_plot["TSNE1"] = tsne_result[:, 0]
    df_plot["TSNE2"] = tsne_result[:, 1]

    plt.figure(figsize=(9, 7))
    for group, color in combined_color_map.items():
        subset = df_plot[df_plot["Group"] == group]
        plt.scatter(
            subset["TSNE1"], subset["TSNE2"],
            c=color, label=group, s=60, alpha=0.8, edgecolors='k'
        )
    plt.title("t-SNE Visualization – All Groups (With/Without Image × Diagnosis)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Category", loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# --- Create and save three t-SNE plots (each with its own fit) ---
plot_diagnosis(df, "Melanoma", f"{output_dir}/tsne_melanoma_{timestamp}.png")
plot_diagnosis(df, "Nevus", f"{output_dir}/tsne_nevus_{timestamp}.png")
plot_combined(df, f"{output_dir}/tsne_combined_{timestamp}.png")
