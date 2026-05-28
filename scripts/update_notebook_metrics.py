import json
import os

notebook_path = 'c:/Users/Admin/Documents/Windows_codespace/DEMETER/Demeter/notebooks/pipeline_verification.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 5 (Index 5) is the CNN Inference Benchmark code
cnn_source = nb['cells'][5]['source']
new_cnn_source = []
for line in cnn_source:
    if "cnn_times = []" in line:
        new_cnn_source.append("cnn_times = []\n")
        new_cnn_source.append("cnn_prep_times = []\n")
        new_cnn_source.append("cnn_inf_times = []\n")
    elif "    # Inference" in line:
        new_cnn_source.append("    prep_time = time.time()\n")
        new_cnn_source.append(line)
    elif "cnn_times.append" in line:
        new_cnn_source.append(line)
        new_cnn_source.append("    cnn_prep_times.append((prep_time - start_time) * 1000)\n")
        new_cnn_source.append("    cnn_inf_times.append((end_time - prep_time) * 1000)\n")
    elif "cnn_avg_time = np.mean(cnn_times) if cnn_times else 0" in line:
        new_cnn_source.append(line)
        new_cnn_source.append("cnn_avg_prep = np.mean(cnn_prep_times) if cnn_prep_times else 0\n")
        new_cnn_source.append("cnn_avg_inf = np.mean(cnn_inf_times) if cnn_inf_times else 0\n")
    elif 'print(f"Average CNN Inference Time:' in line:
        new_cnn_source.append("print(f\"Average CNN Total Time: {cnn_avg_time:.2f} ms per image\")\n")
        new_cnn_source.append("print(f\"  - Preprocessing Time: {cnn_avg_prep:.2f} ms\")\n")
        new_cnn_source.append("print(f\"  - Inference Time: {cnn_avg_inf:.2f} ms\")\n")
    else:
        new_cnn_source.append(line)

nb['cells'][5]['source'] = new_cnn_source

# Cell 7 (Index 7) is the SVM Inference Benchmark code
svm_source = nb['cells'][7]['source']
new_svm_source = []
for line in svm_source:
    if "svm_times = []" in line:
        new_svm_source.append("svm_times = []\n")
        new_svm_source.append("svm_prep_times = []\n")
        new_svm_source.append("svm_inf_times = []\n")
    elif "        svm_model.predict(X_hybrid)" in line:
        new_svm_source.append("        prep_time = time.time()\n")
        new_svm_source.append(line)
    elif "svm_times.append" in line:
        new_svm_source.append(line)
        new_svm_source.append("        svm_prep_times.append((prep_time - start_time) * 1000)\n")
        new_svm_source.append("        svm_inf_times.append((end_time - prep_time) * 1000)\n")
    elif "svm_avg_time = np.mean(svm_times) if svm_times else 0" in line:
        new_svm_source.append(line)
        new_svm_source.append("svm_avg_prep = np.mean(svm_prep_times) if svm_prep_times else 0\n")
        new_svm_source.append("svm_avg_inf = np.mean(svm_inf_times) if svm_inf_times else 0\n")
    elif 'print(f"Average SVM Pipeline Time:' in line:
        new_svm_source.append("print(f\"Average SVM Total Time: {svm_avg_time:.2f} ms per image\")\n")
        new_svm_source.append("print(f\"  - Preprocessing Time (Otsu + FFT + HSV): {svm_avg_prep:.2f} ms\")\n")
        new_svm_source.append("print(f\"  - Inference Time: {svm_avg_inf:.2f} ms\")\n")
    else:
        new_svm_source.append(line)

nb['cells'][7]['source'] = new_svm_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
