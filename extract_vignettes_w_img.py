# extract vignettes with images
import re
import json
from collections import Counter
from docx import Document

# === Load DOCX ===
file_path = "/mnt/d/Naved/Data/derm_vignette/Supplementary Text 1_10132025.docx"
output_file = "/mnt/d/Naved/Data/derm_vignette/Supplementary_Text_1_10132025.json"

doc = Document(file_path)

vignettes = []

# Loop through all tables
for table in doc.tables:
    header_text = table.cell(0, 0).text.strip()

    # Match Case header like "Case #01 (ISIC_0024235) Basal cell carcinoma"
    m = re.match(r"Case\s*#(\d+)\s*\(ISIC[_\s]*(\d+)\)\s*(.*)", header_text, re.IGNORECASE)
    if not m:
        continue

    case_num = int(m.group(1))
    isic_id = m.group(2)
    diagnosis = m.group(3).strip()

    # Find vignette text (before the question)
    vignette_text = ""
    for r in table.rows:
        for c in r.cells:
            txt = c.text.strip()
            if "What is the most likely diagnosis" in txt:
                vignette_text = txt.split("What is the most likely diagnosis")[0].strip()
                break
        if vignette_text:
            break

    vignette_text = re.sub(r"\s+", " ", vignette_text)

    if vignette_text:
        vignettes.append({
            "case_number": case_num,
            "isic_id": isic_id,
            "diagnosis": diagnosis,
            "vignette": vignette_text
        })

# === Save all vignettes to JSON ===

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(vignettes, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(vignettes)} vignettes and saved to '{output_file}'")

# === Compute frequency of each diagnosis ===
diagnoses = [v["diagnosis"] for v in vignettes]
diagnosis_counts = Counter(diagnoses)

print("\n📊 Diagnosis Frequency:")
for diag, count in diagnosis_counts.items():
    print(f"  {diag}: {count}")

# === Optionally, save frequency summary to a separate JSON ===
# freq_output = "/mnt/d/Naved/Data/derm_vignette/diagnosis_frequency.json"
# with open(freq_output, "w", encoding="utf-8") as f:
    # json.dump(diagnosis_counts, f, indent=2, ensure_ascii=False)

# print(f"\n✅ Saved frequency summary to '{freq_output}'")

# === Sanity check: print first 3 vignettes ===
# for v in vignettes[:3]:
    # print(f"\nCase #{v['case_number']} ({v['diagnosis']}):\n{v['vignette'][:200]}...\n")
