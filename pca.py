import numpy as np
import matplotlib.pyplot as plt

def pca_superpixel_embeddings(cluster_centers, dim=2):
    X = cluster_centers.copy().astype(np.float64)

    # standardize
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std == 0] = 1

    X_std = (X - mean) / std

    # center the data
    X_centered = X_std - np.mean(X_std, axis=0)

    # covariance matrix
    cov_mat = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    # get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    sorted_idx = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # get top n vectors
    if dim < 2:
        raise ValueError("Atleast 2 dimensions required")
    elif dim > X.shape[1]:
        raise ValueError(f"{dim} dimensions is greater than available features")
    else:
        W = eigenvectors[:, :dim]
    
    # project the data
    Z = np.dot(X_centered, W)

    # plot
    if dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            Z[:, 0],
            Z[:, 1],
            c=np.arange(len(cluster_centers)),
            cmap='tab20',
            s=60
        )

        plt.title("PCA of Superpixel Centers (from scratch)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)
        plt.show()

    elif dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            Z[:, 0],
            Z[:, 1],
            Z[:, 2],
            c=np.arange(len(cluster_centers)),
            cmap='tab20',
            s=60
        )

        ax.set_title("PCA of Superpixel Centers (from scratch)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        plt.show()

    else:
        print(f"PCA computed for {dim} dimensions. Plotting only supported for 2D or 3D.")

    # explained variance
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    print("Explained variance ratio:")
    for i in range(dim):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f}")

    return Z