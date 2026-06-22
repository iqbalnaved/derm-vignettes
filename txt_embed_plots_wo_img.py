import json
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from datetime import datetime
from huggingface_hub import login
import os 

# --- (Optional) Hugging Face login ---
login(os.environ['HF_TOKEN')

# --- Load ONLY "Without image" vignettes ---
def load_vignettes(file_path, source_label="Without image"):
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

# JSON for "without image"
vignettes, labels, sources = load_vignettes(
    "/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_3_11032025_highlighted_text.json"
)

print(f"Loaded {len(vignettes)} 'without image' vignettes.")

# --- Load Med-GEMMA model ---
model_name = "google/medgemma-4b-pt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# --- Convert vignettes to embeddings ---
def get_embeddings(texts, batch_size=4):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings).numpy()

print("Generating embeddings...")
embeddings = get_embeddings(vignettes)
print("Embeddings shape:", embeddings.shape)

# --- PCA ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

# --- Combine metadata ---
df = pd.DataFrame({
    "Diagnosis": labels,
    "Source": sources,
    "PCA1": pca_result[:, 0],
    "PCA2": pca_result[:, 1],
    "TSNE1": tsne_result[:, 0],
    "TSNE2": tsne_result[:, 1]
})
df["Group"] = df["Source"] + " – " + df["Diagnosis"]

# --- Keep SAME colors as 4-class legend ---
color_map = {
    "Without image – Melanoma": "#2ECC71",  # green (same as before)
    "Without image – Nevus": "#F1C40F"      # yellow (same as before)
}

# --- Timestamp for filenames ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

# --- Plot helper ---
def plot_scatter(x, y, title, base_filename):
    filename = f"{base_filename}_{timestamp}.png"
    plt.figure(figsize=(9,7))
    for group, color in color_map.items():
        subset = df[df["Group"] == group]
        plt.scatter(subset[x], subset[y], c=color, label=group, s=70, alpha=0.85, edgecolors='k')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title="Category", loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved: {filename}")

# --- Save plots ---
output_dir = "/mnt/d/Naved/Outputs/derm_vignettes"
plot_scatter("PCA1", "PCA2", "PCA Visualization (Without Image – Med-GEMMA-4B-PT)",
             f"{output_dir}/pca_medgemma_4b_without_image")
plot_scatter("TSNE1", "TSNE2", "t-SNE Visualization (Without Image – Med-GEMMA-4B-PT)",
             f"{output_dir}/tsne_medgemma_4b_without_image")
