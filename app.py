import io
import base64
import traceback
import numpy as np
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from slic import slic, visualize
from pca  import pca_superpixel_embeddings

app = Flask(__name__)
CORS(app)

def decode_image(b64_string: str) -> np.ndarray:
    header, data = b64_string.split(",", 1)
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def encode_figure(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return "data:image/png;base64," + b64


def encode_array(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return "data:image/png;base64," + b64

@app.route("/api/slic", methods=["POST"])
def api_slic():
    try:
        body    = request.get_json(force=True)
        image   = decode_image(body["image"])          # H×W×3
        K       = int(body.get("K",    100))
        m       = int(body.get("m",     10))
        n_iter  = int(body.get("iters",  5))

        h, w    = image.shape[:2]

        labels, cluster_centers = slic(image, K=K, m=m, num_iterations=n_iter)

        # visualize() calls plt.show() which we can't use on a server,
        # so we replicate its logic and capture the array instead.
        output = image.copy()
        boundary_color = np.array([255, 60, 60], dtype=np.uint8)
        for y in range(h - 1):
            for x in range(w - 1):
                if (labels[y, x] != labels[y, x + 1] or
                        labels[y, x] != labels[y + 1, x]):
                    output[y, x] = boundary_color

        boundary_b64 = encode_array(output)

        # color-filled view (each superpixel → unique hue)
        filled = np.zeros((h, w, 3), dtype=np.uint8)
        n_centers = len(cluster_centers)
        for idx in range(n_centers):
            hue = (idx * 137.508) % 360
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue/360, 0.65, 0.80)
            filled[labels == idx] = [int(r*255), int(g*255), int(b*255)]
        filled_b64 = encode_array(filled)

        return jsonify({
            "ok":              True,
            "boundary_image":  boundary_b64,
            "filled_image":    filled_b64,
            "n_superpixels":   n_centers,
            "image_size":      [w, h],
            # send centers to frontend so PCA call doesn't need the image again
            "centers":         cluster_centers.tolist(),
        })

    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500

@app.route("/api/pca", methods=["POST"])
def api_pca():
    try:
        body    = request.get_json(force=True)
        centers = np.array(body["centers"], dtype=np.float64)
        dim     = int(body.get("dim", 2))
        # pca_superpixel_embeddings() calls plt.show() internally;
        # we capture the figure instead.
        import matplotlib.pyplot as plt
        original_show = plt.show
        plt.show = lambda *a, **kw: None    # suppress display

        # call with dim so the function's own plotting runs silently
        Z = pca_superpixel_embeddings(centers, dim=dim)

        plt.show = original_show            # restore
        colors = np.arange(len(centers))
        evs    = _explained_variance(centers, dim)

        if dim == 2:
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor("#0a0a0f")
            ax.set_facecolor("#12121a")
            sc = ax.scatter(Z[:, 0], Z[:, 1],
                            c=colors, cmap="plasma", s=40, alpha=0.85)
            ax.set_xlabel(f"PC1  ({evs[0]*100:.1f}% var)",
                          color="#7a7a99", fontsize=10)
            ax.set_ylabel(f"PC2  ({evs[1]*100:.1f}% var)",
                          color="#7a7a99", fontsize=10)
            ax.set_title("PCA of Superpixel Centers",
                         color="#f0eeff", fontsize=12, pad=10)
            ax.tick_params(colors="#3a3a55")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1a1a26")
            ax.grid(True, color="#1a1a26", linewidth=0.8)
            plt.colorbar(sc, ax=ax, label="Superpixel index").ax.yaxis.label.set_color("#7a7a99")

        elif dim == 3:
            fig = plt.figure(figsize=(8, 6))
            fig.patch.set_facecolor("#0a0a0f")
            ax  = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#12121a")
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2],
                       c=colors, cmap="plasma", s=40, alpha=0.85)
            ax.set_xlabel(f"PC1 ({evs[0]*100:.1f}%)", color="#7a7a99", fontsize=9)
            ax.set_ylabel(f"PC2 ({evs[1]*100:.1f}%)", color="#7a7a99", fontsize=9)
            ax.set_zlabel(f"PC3 ({evs[2]*100:.1f}%)", color="#7a7a99", fontsize=9)
            ax.set_title("PCA of Superpixel Centers",
                         color="#f0eeff", fontsize=12, pad=10)

        pca_b64 = encode_figure(fig)

        return jsonify({
            "ok":        True,
            "pca_image": pca_b64,
            "Z":         Z.tolist(),
            "explained": [float(e) for e in evs],
        })

    except Exception:
        return jsonify({"ok": False, "error": traceback.format_exc()}), 500

def _explained_variance(centers: np.ndarray, dim: int):
    """Mirror the eigenvalue logic in pca.py to get explained variance ratios."""
    X = centers.copy().astype(np.float64)
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0)
    std[std == 0] = 1
    X_std = (X - mean) / std
    X_c   = X_std - np.mean(X_std, axis=0)
    cov   = np.dot(X_c.T, X_c) / (X_c.shape[0] - 1)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    return eigenvalues[:dim] / np.sum(eigenvalues)

# run

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    print("Starting SLIC + PCA API server on http://localhost:5000")
    app.run(debug=True, port=5000)