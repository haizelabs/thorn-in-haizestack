import glob
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import re
from matplotlib.colors import LinearSegmentedColormap

json_files = glob.glob(f"scores/*.json")
if not os.path.exists("viz/"):
    os.makedirs("viz/")

for file in json_files:
    data = []
    with open(file, 'r') as f:
        json_data = json.load(f)
        for datum in json_data:
            document_depth = datum.get("loc", None)
            context_length = datum.get("ctx_len", None)
            score = datum.get("score", None)
            data.append({
                "Attack Location": document_depth,
                "Context Length": context_length,
                "Safety Score": score
            })

    df = pd.DataFrame(data)
    df['Context Length'] = df['Context Length'].astype(int)

    pivot_table = pd.pivot_table(df, values='Safety Score', index=['Attack Location', 'Context Length'], aggfunc='mean').reset_index()
    pivot_table = pivot_table.pivot(index="Attack Location", columns="Context Length", values="Safety Score")

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#4C9949", "#C7C3C1", "#741A29"])
    plt.figure(figsize=(17.5, 8))
    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Response Harmfulness'}
    )

    model_name = re.search(r"/([^/]+?)\.json", file).group(1)
    if model_name == "openai": model_name = "GPT-4"
    if model_name == "anthropic": model_name = "Claude"
    if model_name == "cohere": model_name = "Command-R"
    
    plt.title(f'Thorn in a HaizeStack\nJailbreaking {model_name} by Inserting a Simple Attack in a Sea of Text')
    plt.xlabel('Context Length')
    plt.ylabel('Jailbreak Location')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"viz/{model_name}.png")