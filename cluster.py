import numpy as np
import os
import pandas as pd

import umap.umap_ as umap
import matplotlib.pyplot as plt

import pickle 
import plotly.graph_objects as go

emb_path = r"F:\Workspace\bioshock-audio-clustering\all_voicelines_emb"


# Load embeddings and create index map
# ================================
embeddings = []

emb_files = [os.path.join(emb_path, file) for file in os.listdir(emb_path)]

index = 0
index_map = {}

for file in emb_files:
    emb = np.load(file)
    emb = emb.flatten()
    
    if emb.shape[0] == 192:
        embeddings.append(emb)
        index_map[index] = file.split("\\")[-1].replace(".npy", "")
        index += 1

with open('./index_map.pkl', 'wb') as f:
    pickle.dump(index_map, f)

np.save("all_embeddings.npy", np.array(embeddings))
# ================================


# Load embeddings and index map
embeddings = np.load("./all_embeddings.npy")
with open('./index_map.pkl', 'rb') as f:
    index_map = pickle.load(f)


print(embeddings.shape)
print(len(index_map))

audio_files = [os.path.join(r"F:\Workspace\bioshock-audio-clustering\all_voicelines_wav", index_map[i] + ".wav") for i in range(len(index_map.keys()))]
print(audio_files[:5])

reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, metric="euclidean")
embedding_2d = reducer.fit_transform(np.array(embeddings))


df = pd.DataFrame({
    "UMAP1": embedding_2d[:, 0][:1],
    "UMAP2": embedding_2d[:, 1][:1],
    "audio_files": audio_files[:1]
})


# Create scatter plot with hover tooltips
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["UMAP1"],
    y=df["UMAP2"],
    mode="markers",
    marker=dict(size=10, color='blue'),
    text=df["audio_files"],  # Store file paths in the text field
    hoverinfo="text"  # Show filename when hovering
))

# Add axis labels
fig.update_layout(
    title="UMAP Clustering (Hover for Filepath)",
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2"
)
# Show the plot
# fig.show()



from IPython.display import display, HTML
import pandas as pd
import plotly.graph_objects as go

df = pd.DataFrame({
    "UMAP1": embedding_2d[:, 0],
    "UMAP2": embedding_2d[:, 1],
    "audio_files": audio_files
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["UMAP1"],
    y=df["UMAP2"],
    mode="markers",
    marker=dict(size=10, color='blue'),
    text=df["audio_files"],
    hoverinfo="text",
    customdata=df["audio_files"]
))

fig.update_layout(
    title="UMAP Clustering (Click to open audio file)",
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2"
)

# Display the figure with custom JavaScript
display(HTML('''
<div id="plot"></div>
<script>
require(["plotly"], function(Plotly) {
    // Display the plot
    Plotly.newPlot('plot', %s, %s);
    
    // Handle click events
    document.getElementById('plot').on('plotly_click', function(data) {
        var filePath = data.points[0].customdata;
        window.open(filePath);
    });
});
</script>
''' % (fig.to_json(), fig.layout.to_json())))




# plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)
# plt.title("UMAP Projection of Audio Features")
# plt.show()





# tsne plot
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, random_state=0)
# embedding_2d = tsne.fit_transform(np.array(embeddings))

# plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)
# plt.title("TSNE Projection of Audio Features")
# plt.show()


# # clustering with DBSCAN
# from sklearn.cluster import DBSCAN

# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan.fit(embedding_2d)

# plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=dbscan.labels_, alpha=0.7)
# plt.title("DBSCAN Clustering of Audio Features")
# plt.show()