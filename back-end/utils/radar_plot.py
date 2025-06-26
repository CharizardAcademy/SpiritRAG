import matplotlib.pyplot as plt
import numpy as np

# Evaluation dimensions
retriever_labels = ['Relevance', 'Accuracy', 'Usefulness', 'Temporality', 'Actionability']
generator_labels = ['Congruence', 'Coherence', 'Relevance', 'Creativity', 'Engagement']

# Scores (no rescaling)
retriever_models_scaled = {
    'all-MiniLM-L6-v2': [3.78, 4.45, 3.66, 4.45, 2.57],
    'Qwen3-Embedding-0.6B': [4.26, 4.99, 3.89, 4.99, 2.08]
}
generator_models_scaled = {
    'Qwen3-0.6B': [2.27, 3.29, 2.44, 1.94, 2.06],
    'Qwen3-1.7B': [3.41, 3.94, 4.18, 3.39, 2.51]
}

# Function to create radar chart
def create_compact_radar(ax, labels, model_data, colors, title):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor('#f5f5f5')

    handles, labels_out = [], []
    for (model, values), color in zip(model_data.items(), colors):
        values_extended = values + [values[0]]
        line, = ax.plot(angles, values_extended, color=color, linewidth=1.5, marker='o', markersize=6, label=model)
        ax.fill(angles, values_extended, color=color, alpha=0.4)
        handles.append(line)
        labels_out.append(model)

    # ax.set_yticks([1, 2, 3, 4, 5])
    # ax.set_ylim(0, 5)

    
    for angle in angles:
        ax.plot([angle, angle], [0, 5], color='gray', linestyle='--', linewidth=1.5, zorder=0)
        
    ax.grid(True, color='gray', linestyle='--', linewidth=1.5)
    ax.set_xticks([])
    #ax.set_yticklabels(['1', '2', '3', '4', '5'], color='black', fontsize=14, zorder=100)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14, pad=20)
    return handles, labels_out

# Create figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(6, 6), subplot_kw=dict(polar=True))

# Radar plots
handles1, labels1 = create_compact_radar(axs[0], retriever_labels, retriever_models_scaled, ['#2F8B57', '#097969'], "Document Retrieval")


handles2, labels2 = create_compact_radar(axs[1], generator_labels, generator_models_scaled, ['#6856F2', '#40389E'], "Answer Generation")

angle_deg = 60
angle_rad = np.deg2rad(angle_deg)
for ax in axs:
    for r in [1, 2, 3, 4, 5]:
        x =40
        y = r - 0.05
        ax.text(x, y, str(r), ha='left', va='center', fontsize=14, zorder=100, color="#4D4D4D")

# Add horizontal legend below both plots
fig.legend(handles1 + handles2,
           labels1 + labels2,
           loc='lower center',
           ncol=2,
           fontsize=14,
           frameon=True,
           bbox_to_anchor=(0.5, 0.07))

# Adjust layout
plt.subplots_adjust(top=0.9, bottom=0, left=0.05, right=0.95, wspace=0.3)
plt.savefig("teaser.png", dpi=300, pad_inches=0.05)
plt.show()

