import umap
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
def cluster_plot(data,device,imagename,n_clsuters):
    neighbor=18
    batch_size = 100
    max_iter = 100
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=max_iter)
    mbkmeans.fit(data)
    cluster_assignments = mbkmeans.labels_
    cluster_centers = mbkmeans.cluster_centers_
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(data)
    cluster_labels = mbkmeans.labels_
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    num_classes = n_clusters
    cmap = ListedColormap(colors[:num_classes])
    plt.figure(figsize=(18, 16))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=cluster_labels, cmap=cmap, s=5)
    plt.colorbar()
    plt.title("UMAP Visualization of Mini-Batch K-Means Clusters")
    plt.show()
    plt.savefig('cluster.png',bbox_inches='tight')
    return(cluster_labels)

