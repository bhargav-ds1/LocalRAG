import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns

files = glob.glob('Evaluation/EvaluationResults/*/*.csv')
whole_df = []
for file in files:
    df = pd.read_csv(file)
    df.columns = ['metrics', file.split('/')[-2]]
    df = df.set_index('metrics').unstack().unstack()
    whole_df.append(df)
whole_df = pd.concat(whole_df)

cmap = plt.get_cmap('viridis', len(whole_df.index))

category_colors = {cat: cmap(i) for i, cat in enumerate(whole_df.index.unique())}


fig, axs = plt.subplots(1, 3, figsize = (18,6), sharey=True)

for ax, column in zip(axs, ['mean_relevancy_score', 'mean_faithfulness_score', 'mean_context_similarity_score']):
    bars = ax.bar(whole_df.index, whole_df[column], color=[category_colors[cat] for cat in whole_df.index])
    ax.set_title(column)
    # Fixed the tick labeling method to avoid the previous warning
    ax.set_xticks([])

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=category_colors[cat], label=cat) for cat in category_colors]
fig.legend(handles=legend_elements, title="Categories")
plt.tight_layout()
plt.show()