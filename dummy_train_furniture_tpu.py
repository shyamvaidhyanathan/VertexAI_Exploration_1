# /root/train_furniture_tpu.py
import os
import argparse
import tensorflow as tf

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gcs_data", required=True)
    p.add_argument("--classes", required=True)  # e.g. "chair,sofa,table,bed"
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--device", choices=["auto", "tpu", "cpu"], default="auto",  help="Training device preference. 'auto' tries TPU then falls back.")
    return p.parse_args()

def get_strategy(device_choice="auto"):
    print(f"[trainer] device_choice={device_choice}")
    if device_choice == "cpu":
        print("[trainer] Forcing CPU/GPU strategy.")
        return tf.distribute.get_strategy()

    # Try TPU, otherwise fall back
    try:
        # This auto-detects TPUs on Vertex TPU VMs; if none, it will raise.
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        print("[trainer] ✅ Connected to TPU.")
        print("[trainer] TPU devices:", tf.config.list_logical_devices("TPU"))
        return tf.distribute.TPUStrategy(resolver)
    except Exception as e:
        if device_choice == "tpu":
            raise RuntimeError(f"TPU requested but not available: {e}") from e
        print("[trainer] ⚠️ TPU not available; falling back to CPU/GPU. Reason:", e)
        return tf.distribute.get_strategy()



def make_dummy_dataset(batch_size: int, steps: int, num_classes: int):
    # Synthetic data so the pipeline runs even if GCS data is empty while debugging.
    import numpy as np
    def gen():
        while True:
            x = np.random.randint(0, 255, size=(batch_size, 128, 128, 3), dtype="uint8")
            y = np.random.randint(0, num_classes, size=(batch_size,), dtype="int32")
            yield x, y
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 128, 128, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        ),
    ).take(5)
    return ds



def _list_all_images(gcs_data: str, classes: list[str]) -> list[str]:
    """Collect filepaths for jpg/jpeg/png in each class subdir."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for cls in classes:
        for pat in patterns:
            files += tf.io.gfile.glob(f"{gcs_data}/{cls}/{pat}")
    return files

def _build_label_lookup(classes: list[str]) -> tf.lookup.StaticHashTable:
    keys = tf.constant(classes)
    vals = tf.constant(list(range(len(classes))), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1
    )
    return table

def _path_to_example(path: tf.Tensor, table, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """path -> (image[H,W,3] float32 [0,1], label int64)"""
    # label from parent folder name (…/<label>/filename.jpg)
    label_name = tf.strings.split(path, "/")[-2]
    label = table.lookup(label_name)

    # load & decode
    img_bytes = tf.io.read_file(path)
    # decode_image handles JPG/PNG; expand_animations=False avoids GIF frames
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def build_datasets(
    gcs_data: str,
    classes_csv: str,
    img_size: int = 128,
    batch_size: int = 64,
    train_split: float = 0.8,
    shuffle_buffer: int = 2000,
    using_tpu: bool = False,
):
    """
    Returns (train_ds, val_ds, class_names). Both are batched & prefetched.
    Expects directory layout: <gcs_data>/<class_name>/*.{jpg,jpeg,png}
    """
    class_names = [c.strip() for c in classes_csv.split(",") if c.strip()]
    if len(class_names) < 2:
        raise ValueError("Provide at least 2 classes via --classes")

    # Python-side file listing (robust split)
    filepaths = _list_all_images(gcs_data, class_names)
    if len(filepaths) == 0:
        raise ValueError(f"No images found under {gcs_data}. Check bucket & paths.")

    # shuffle deterministically for reproducibility
    rng = np.random.default_rng(seed=42)
    rng.shuffle(filepaths)

    n_total = len(filepaths)
    n_train = max(1, int(n_total * float(train_split)))
    train_files = filepaths[:n_train]
    val_files   = filepaths[n_train:] if n_total > n_train else filepaths[-1:]

    print(f"[data] total={n_total}  train={len(train_files)}  val={len(val_files)}")
    for cls in class_names:
        cnt = sum(1 for p in filepaths if f"/{cls}/" in p)
        print(f"[data] {cls:>10s}: {cnt} files")

    # TF lookup table for label names -> ids
    table = _build_label_lookup(class_names)

    def make_ds(paths: list[str]) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        # shuffle file order (buffered). For validation we can skip big shuffles.
        ds = ds.shuffle(min(len(paths), shuffle_buffer), reshuffle_each_iteration=True)
        ds = ds.map(lambda p: _path_to_example(p, table, img_size), num_parallel_calls=AUTOTUNE)
        # filter out any unknown labels (-1), just in case
        ds = ds.filter(lambda x, y: tf.greater_equal(y, tf.cast(0, y.dtype)))
        # (optional) light augmentation on train only
        return ds

    train_ds = make_ds(train_files)
    val_ds   = make_ds(val_files)

    # Augment train a bit
    def aug(x, y):
        x = tf.image.random_flip_left_right(x)
        return x, y
    train_ds = train_ds.map(aug, num_parallel_calls=AUTOTUNE)

    # Batch/Prefetch
    drop = True if using_tpu else False  # TPUs prefer static shapes
    train_ds = train_ds.batch(batch_size, drop_remainder=drop).prefetch(AUTOTUNE)
    val_ds   = val_ds.batch(batch_size, drop_remainder=drop).prefetch(AUTOTUNE)

    # Set AutoShardPolicy (safe with or without TPU)
    opts = tf.data.Options()
    opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.with_options(opts)
    val_ds   = val_ds.with_options(opts)

    return train_ds, val_ds, class_names











def main():
    args = parse_args()
    print("[trainer] TF:", tf.__version__)
    print("[trainer] gcs_data:", args.gcs_data)
    print("[trainer] classes:", args.classes)
    print("[trainer] epochs:", args.epochs, "batch_size:", args.batch_size)
    print("[trainer] model_dir:", args.model_dir)

    labels = [c.strip() for c in args.classes.split(",") if c.strip()]
    num_classes = max(2, len(labels))  # avoid degenerate 1-class
    print("[trainer] parsed labels:", labels, "num_classes:", num_classes)

    # detect TPU to decide drop_remainder
    using_tpu = isinstance(strategy, tf.distribute.TPUStrategy)

    train_ds = make_dummy_dataset(args.batch_size, steps=5, num_classes=num_classes)
    model.fit(train_ds, epochs=max(1, args.epochs))


    print("Cardinality train:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Cardinality val  :", tf.data.experimental.cardinality(val_ds).numpy())
    for x, y in train_ds.take(1):
        print("Sample batch shape:", x.shape, "labels example:", y[:8].numpy())




    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(1, args.epochs),
    )



    export_dir = os.path.join(args.model_dir, "saved_model")
    print("[trainer] Exporting SavedModel to:", export_dir)
    tf.saved_model.save(model, export_dir)
    print("[trainer] ✅ Export complete.")

if __name__ == "__main__":
    main()
