import json
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from datetime import datetimetalkbots
reportsresponse
from huggingface_hub import login

login(os.environ['HF_TOKEN'])

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

# Two files
v_img, l_img, s_img = load_vignettes("/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_1_10132025.json", "With image")
v_txt, l_txt, s_txt = load_vignettes("/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_3_10312025.json", "Without image")

vignettes = v_img + v_txt
labels = l_img + l_txt
sources = s_img + s_txt

print(f"Loaded {len(vignettes)} total vignettes.")

# --- Load Med-GEMMA model ---
model_name = "google/medgemma-4b-pt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# --- Convert vignettes to embeddings ---
def get_embeddings(texts):
    all_embeddings = []
    for i in range(0, len(texts), 4):  # batch for efficiency
        batch = texts[i:i+4]
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
    "PCA1": pca_result[:,0],
    "PCA2": pca_result[:,1],
    "TSNE1": tsne_result[:,0],
    "TSNE2": tsne_result[:,1]
})
df["Group"] = df["Source"] + " – " + df["Diagnosis"]

# --- Color map for 4 groups ---
color_map = {
    "With image – Melanoma": "#E74C3C",     # red
    "With image – Nevus": "#3498DB",        # blue
    "Without image – Melanoma": "#2ECC71",  # green
    "Without image – Nevus": "#F1C40F"      # yellow
}

# --- Timestamp for filenames ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

# --- Plot helper ---
def plot_scatter(x, y, title, filename):
    plt.figure(figsize=(9,7))
    for group, color in color_map.items():
        subset = df[df["Group"] == group]
        plt.scatter(subset[x], subset[y], c=color, label=group, s=60, alpha=0.8, edgecolors='k')
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title="Category", loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# --- Save plots ---
plot_scatter("PCA1", "PCA2", "PCA Visualization (Med-GEMMA-4B-PT)", f"/mnt/d/Naved/Outputs/derm_vignettes/pca_medgemma_4b_{timestamp}.png")
plot_scatter("TSNE1", "TSNE2", "t-SNE Visualization (Med-GEMMA-4B-PT)", f"/mnt/d/Naved/Outputs/derm_vignettes/tsne_medgemma_4b_{timestamp}.png")
