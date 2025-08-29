# test_predict.py
import argparse
import os
import numpy as np
from PIL import Image
from google.cloud import aiplatform


def load_image(path: str, size=(128, 128)) -> np.ndarray:
    """Load an image, resize, normalize to [0,1], return shape (128,128,3) float32."""
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr  # (H, W, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="Endpoint region")
    parser.add_argument("--endpoint_id", required=True, help="Vertex AI Endpoint ID")
    parser.add_argument("--image", required=True, help="Path to local image file")
    parser.add_argument(
        "--classes",
        default="chair,sofa,table,bed",
        help="Comma-separated class labels in the same order used for training",
    )
    args = parser.parse_args()

    labels = [c.strip() for c in args.classes.split(",") if c.strip()]
    if len(labels) < 2:
        raise ValueError("You must provide at least 2 class labels via --classes")

    # 1) Preprocess image -> instance for prediction
    arr = load_image(args.image)                 # (128,128,3), float32 in [0,1]
    instance = arr.tolist()                      # JSON-serializable

    # 2) Init Vertex AI & get endpoint handle
    aiplatform.init(project=args.project_id, location=args.region)
    endpoint = aiplatform.Endpoint(endpoint_name=args.endpoint_id)

    # 3) Predict
    resp = endpoint.predict(instances=[instance])
    preds = np.array(resp.predictions)           # shape (1, num_classes) for our softmax model
    if preds.ndim != 2 or preds.shape[0] != 1:
        raise RuntimeError(f"Unexpected predictions shape: {preds.shape}")

    probs = preds[0]
    if len(probs) != len(labels):
        print("⚠️ Warning: number of output scores doesn't match labels count.")
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_prob = float(probs[top_idx])

    # 4) Pretty print
    print("\n--- Prediction ---")
    print(f"Image: {args.image}")
    print(f"Endpoint: {args.endpoint_id} (region {args.region})")
    print(f"Top-1: {top_label}  (p={top_prob:.3f})\n")
    print("All class probabilities:")
    for i, p in enumerate(probs):
        lab = labels[i] if i < len(labels) else f"class_{i}"
        print(f"  {lab:>10s} : {p:.3f}")


if __name__ == "__main__":
    # Make sure you’re authenticated:
    #   gcloud auth application-default login
    # The active identity needs roles/aiplatform.user on the project.
    main()
