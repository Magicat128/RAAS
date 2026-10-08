import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

with open("data/llama-8b_trans_data.json", "r") as f:
    data = json.load(f)
    data1, data2 = data["sim_correct"], data["sim_not_correct"]

my_palette = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#17becf", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

color_map = {
    "Initial\nQuestion": my_palette[0],
    "Problem\nSetup": my_palette[1],
    "Plan\nGeneration": my_palette[2],
    "Fact\nRetrieval": my_palette[3],
    "Active\nComputation": my_palette[4],
    "Result\nConsol.": my_palette[5],
    "Uncertainty\nMgmt.": my_palette[6],
    "Self\nChecking": my_palette[7],
    "Final Answer\nEmission": my_palette[8]
}

df1 = pd.DataFrame(data1).T
df2 = pd.DataFrame(data2).T
diff = df2 - df1

diff = diff.rename(columns={
    "initial_question": "Initial\nQuestion",
    "problem_setup": "Problem\nSetup",
    "plan_generation": "Plan\nGeneration",
    "fact_retrieval": "Fact\nRetrieval",
    "active_computation": "Active\nComputation",
    "result_consolidation": "Result\nConsol.",
    "uncertainty_management": "Uncertainty\nMgmt.",
    "self_checking": "Self\nChecking",
    "final_answer_emission": "Final Answer\nEmission"
})
diff = diff.rename(index={
    "initial_question": "Initial\nQuestion",
    "problem_setup": "Problem\nSetup",
    "plan_generation": "Plan\nGeneration",
    "fact_retrieval": "Fact\nRetrieval",
    "active_computation": "Active\nComputation",
    "result_consolidation": "Result\nConsol.",
    "uncertainty_management": "Uncertainty\nMgmt.",
    "self_checking": "Self\nChecking",
    "final_answer_emission": "Final Answer\nEmission"
})

plt.figure(figsize=(14, 10))
sns.heatmap(diff, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=.5, cbar_kws={'shrink': 0.5})
plt.title("Differences between steered transition and original transition (without intervention)", fontsize=16)
plt.xlabel("To Category (Step t)", fontsize=16)
plt.ylabel("From Category (Step t-1)", fontsize=16)
#plt.xticks(rotation=45)

ax = plt.gca()
ax.set_aspect(0.5)
plt.yticks(rotation=0)

for label in ax.get_yticklabels():
    text = label.get_text()
    color = color_map.get(text, 'black')
    label.set_color(color)

for label in ax.get_xticklabels():
    text = label.get_text()
    color = color_map.get(text, 'black')
    label.set_color(color)

for i in range(diff.shape[0]):
    max_col = diff.iloc[i].idxmax()
    min_col = diff.iloc[i].idxmin()

    max_col_idx = diff.columns.get_loc(max_col)
    min_col_idx = diff.columns.get_loc(min_col)

    ax.add_patch(patches.Rectangle(
        (max_col_idx, i), 1, 1, fill=False, edgecolor='magenta', lw=3
    ))
    ax.add_patch(patches.Rectangle(
        (min_col_idx, i), 1, 1, fill=False, edgecolor='#00FFFF', lw=3
    ))

plt.tight_layout()
plt.savefig('figure.png')
