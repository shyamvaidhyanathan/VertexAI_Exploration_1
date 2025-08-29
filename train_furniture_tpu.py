#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transfer-learning trainer using MobileNetV2 (ImageNet) on Vertex AI or local.
- Expects images laid out like:
    gs://<BUCKET>/data/<class_name>/*.{jpg,jpeg,png}
- Uses two training stages:
    1) Freeze the MobileNetV2 backbone; train new classification head
    2) Unfreeze top N layers; fine-tune at a lower learning rate
- Auto-detects TPU; falls back to CPU/GPU. You can force device via --device
- Exports SavedModel to <model_dir>/saved_model and writes class_names.json
"""

import os
import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
np.random.seed(42)
tf.random.set_seed(42)


# --------------------------- CLI ---------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MobileNetV2 transfer-learning trainer")
    p.add_argument("--gcs_data", required=True,  help="GCS prefix with class subfolders, e.g. gs://my-bucket/data")
    p.add_argument("--classes", required=True,   help="Comma-separated class names matching subfolder names, ""e.g. 'almirah,chair,fridge,table,tv'")
    p.add_argument("--model_dir", required=True,help="GCS path for outputs; SavedModel goes to <model_dir>/saved_model")
    p.add_argument("--img_size", type=int, default=224, help="Input size (MobileNetV2 default is 224)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--train_split", type=float, default=0.85,help="Per-class split; rest used for validation")
    p.add_argument("--epochs", type=int, default=20,help="Total epochs (Stage1 + Stage2)")
    p.add_argument("--initial_epochs", type=int, default=5,  help="Stage 1 epochs with backbone frozen")
    p.add_argument("--fine_tune_layers", type=int, default=40,help="Unfreeze this many layers from the END for fine-tuning")
    p.add_argument("--device", choices=["auto", "tpu", "cpu"], default="auto", help="Where to train: auto (TPU if available), tpu, or cpu")
    args, _unknown = p.parse_known_args()
    return args


# ----------------------- Device / Strategy -----------------------------------
def get_strategy(device_choice: str):
    print(f"[trainer] device_choice={device_choice}")
    if device_choice == "cpu":
        print("[trainer] Using default CPU/GPU strategy.")
        return tf.distribute.get_strategy(), False
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        print("[trainer] ✅ Connected to TPU.")
        strategy = tf.distribute.TPUStrategy(resolver)
        return strategy, True
    except Exception as e:
        if device_choice == "tpu":
            raise RuntimeError(f"TPU requested but not available: {e}") from e
        print("[trainer] ⚠️ TPU not available; falling back to CPU/GPU. Reason:", e)
        return tf.distribute.get_strategy(), False


# ----------------------- Data pipeline ---------------------------------------
def _list_all_images(gcs_data: str, classes: List[str]) -> List[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for cls in classes:
        for pat in patterns:
            files += tf.io.gfile.glob(f"{gcs_data.rstrip('/')}/{cls}/{pat}")
    return files


def _build_label_lookup(classes: List[str]) -> tf.lookup.StaticHashTable:
    keys = tf.constant(classes)
    vals = tf.constant(list(range(len(classes))), dtype=tf.int64)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=-1
    )


def _path_to_example(path: tf.Tensor, table, img_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    # Label = folder name (second-to-last component)
    label_name = tf.strings.split(path, "/")[-2]
    label = table.lookup(label_name)

    # Decode & resize
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0,1]
    return img, label


def build_datasets_and_counts(
    gcs_data: str,
    classes_csv: str,
    img_size: int,
    batch_size: int,
    train_split: float,
    using_tpu: bool,
):
    # Parse classes & verify
    class_names = [c.strip() for c in classes_csv.split(",") if c.strip()]
    if len(class_names) < 2:
        raise ValueError("Please provide at least 2 classes via --classes")

    # List files by class
    filepaths = _list_all_images(gcs_data, class_names)
    if not filepaths:
        raise ValueError(
            f"No images found under {gcs_data}. "
            f"Expect folders: {[f'/{c}/' for c in class_names]}"
        )

    per_class = {c: [] for c in class_names}
    for p in filepaths:
        for c in class_names:
            if f"/{c}/" in p:
                per_class[c].append(p)
                break

    # Per-class shuffle & split (keeps classes balanced across splits)
    rng = np.random.default_rng(42)
    train_files, val_files = [], []
    train_counts = {}
    for c in class_names:
        arr = np.array(per_class[c])
        rng.shuffle(arr)
        n = len(arr)
        n_train = max(1, int(n * float(train_split)))
        train_files.extend(arr[:n_train].tolist())
        # Ensure at least 1 val example per class
        if n > n_train:
            val_files.extend(arr[n_train:].tolist())
        else:
            val_files.append(arr[-1])  # duplicate last one if needed
        train_counts[c] = int(min(n_train, n))

    rng.shuffle(train_files)
    rng.shuffle(val_files)

    print(f"[data] total={len(filepaths)}  train={len(train_files)}  val={len(val_files)}")
    for c in class_names:
        print(f"[data] {c:>12s}: all={len(per_class[c])}  train={train_counts[c]}")

    # Create lookup table once
    table = _build_label_lookup(class_names)

    # Light, safe augmentations
    rnd_rotate = tf.keras.layers.RandomRotation(0.05)
    rnd_zoom   = tf.keras.layers.RandomZoom(0.1)

    def make_ds(paths: List[str], augment: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        if augment:
            ds = ds.shuffle(min(len(paths), 2000), reshuffle_each_iteration=True)

        ds = ds.map(lambda p: _path_to_example(p, table, img_size), num_parallel_calls=AUTOTUNE)

        if augment:
            def aug(x, y):
                x = tf.image.random_flip_left_right(x)
                x = tf.image.random_brightness(x, max_delta=0.08)
                x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
                x = rnd_rotate(x, training=True)
                x = rnd_zoom(x, training=True)
                return x, y
            ds = ds.map(aug, num_parallel_calls=AUTOTUNE)

        # TPU prefers fixed batch (drop last partial batch)
        drop = True if using_tpu else False
        ds = ds.batch(batch_size, drop_remainder=drop).prefetch(AUTOTUNE)

        # Good for distributed input
        opts = tf.data.Options()
        opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        return ds.with_options(opts)

    return (
        make_ds(train_files, augment=True),
        make_ds(val_files, augment=False),
        class_names,
        train_counts,
    )


# ----------------------- Model (MobileNetV2 TL) -------------------------------
def build_transfer_model(num_classes: int, image_size: int = 224):
    """
    MobileNetV2 + classification head.
    Our pipeline outputs images in [0,1]. MobileNetV2 expects [-1,1],
    so we add a Rescaling layer inside the model to keep serving consistent.
    """
    from tensorflow.keras import layers, models, applications

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)  # [0,1] -> [-1,1]

    base = applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(image_size, image_size, 3)
    )
    base.trainable = False  # Stage 1: freeze backbone
    


    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")

    print("[build_transfer_model] mobilenetv2_transfer model and base defined")

    return model, base


# ----------------------- Main -------------------------------------------------
def main():
    args = parse_args()
    print("[trainer] TF:", tf.__version__)
    print("[trainer] gcs_data:", args.gcs_data)
    print("[trainer] classes:", args.classes)
    print("[trainer] epochs:", args.epochs, "batch_size:", args.batch_size)
    print("[trainer] model_dir:", args.model_dir)

    strategy, using_tpu = get_strategy(args.device)

    # Build datasets & counts (used for optional class weights)
    train_ds, val_ds, class_names, train_counts = build_datasets_and_counts(
        gcs_data=args.gcs_data,
        classes_csv=args.classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        train_split=args.train_split,
        using_tpu=using_tpu,
    )
    num_classes = len(class_names)

    # Class weights help when classes are imbalanced
    class_weight = None
    if train_counts:
        total = sum(train_counts.values())
        class_weight = {
            i: total / (len(train_counts) * max(1, train_counts[c]))
            for i, c in enumerate(class_names)
        }
        print("[trainer] class_weight:", class_weight)

    # Build model under the chosen strategy
    with strategy.scope():
        model, base = build_transfer_model(num_classes=num_classes, image_size=args.img_size)

    # Callbacks & loss
    early  = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    loss   = tf.keras.losses.SparseCategoricalCrossentropy()

    # ---------------- Stage 1: train head (backbone frozen) -------------------
    stage1_epochs = min(max(1, args.initial_epochs), args.epochs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    print(f"[trainer] Stage 1 (frozen) epochs={stage1_epochs}")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=[early, reduce],
        class_weight=class_weight,
        verbose=2,
    )

    # ---------------- Stage 2: fine-tune backbone top layers ------------------
    remaining = max(0, args.epochs - len(hist1.history["loss"]))
    if remaining > 0:
        k = max(1, args.fine_tune_layers)  # how many layers from the end to unfreeze
        for layer in base.layers[:-k]:
            layer.trainable = False
        for layer in base.layers[-k:]:
            layer.trainable = True

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=["accuracy"])
        print(f"[trainer] Stage 2 (fine-tune last {k} layers) epochs={remaining}")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining,
            callbacks=[early, reduce],
            class_weight=class_weight,
            verbose=2,
        )

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"[eval] val_accuracy: {val_acc:.4f}  val_loss: {val_loss:.4f}")

    # Export SavedModel (for Vertex AI Model Upload/Endpoint)
    export_dir = os.path.join(args.model_dir.rstrip("/"), "saved_model")
    print("[trainer] Exporting SavedModel to:", export_dir)
    tf.saved_model.save(model, export_dir)
    ok = tf.io.gfile.exists(os.path.join(export_dir, "saved_model.pb"))
    print("[trainer] ✅ Export complete." if ok else "[trainer] ❌ Export missing saved_model.pb")

    # Save class names (handy for clients & debugging)
    labels_path = os.path.join(args.model_dir.rstrip("/"), "class_names.json")
    with tf.io.gfile.GFile(labels_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print("[trainer] Wrote label file:", labels_path)


if __name__ == "__main__":
    main()
