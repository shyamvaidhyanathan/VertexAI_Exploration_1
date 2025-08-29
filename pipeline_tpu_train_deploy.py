# pipeline_tpu_train_deploy.py
# ------------------------------------------------------------
# KFP v2 + google-cloud-pipeline-components v2 version
# Trains a TensorFlow model on Vertex AI (TPU by default),
# verifies export, uploads as an UnmanagedContainerModel, and
# deploys to an Endpoint for online prediction.

from kfp import dsl
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.types import artifact_types


# ---------- (Optional) Lightweight components for early validation ----------

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-storage"]
)
def validate_inputs(gcs_data: str) -> str:
    """Fail fast if no images exist under the provided GCS prefix."""
    from google.cloud import storage
    import re

    if not gcs_data.startswith("gs://"):
        raise ValueError(f"gcs_data must start with gs:// (got {gcs_data})")

    # Split gs://bucket/prefix...
    m = re.match(r"^gs://([^/]+)/(.*)$", gcs_data)
    if not m:
        raise ValueError(f"Invalid GCS URI: {gcs_data}")
    bucket_name, prefix = m.group(1), m.group(2)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Look for common image extensions (quick scan).
    total = 0
    for ext in ("jpg", "jpeg", "png"):
        blobs = client.list_blobs(bucket, prefix=prefix, page_size=100)
        # Count up to a small number; we only need to know "exists"
        for b in blobs:
            # Cheap filter: only count names ending with an image extension
            name = b.name.lower()
            if name.endswith(f".{ext}"):
                total += 1
                if total >= 1:
                    break
        if total >= 1:
            break

    if total == 0:
        raise RuntimeError(f"No images found under {gcs_data}. "
                           f"Ensure gs://bucket/class_name/your_image.jpg exists.")
    return f"Found images under {gcs_data}"


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["google-cloud-storage"]
)
def verify_export(model_dir: str) -> str:
    """Check that a SavedModel exists at {model_dir}/saved_model/."""
    from google.cloud import storage
    import re

    if not model_dir.startswith("gs://"):
        raise ValueError(f"model_dir must start with gs:// (got {model_dir})")

    m = re.match(r"^gs://([^/]+)/(.*)$", model_dir)
    if not m:
        raise ValueError(f"Invalid GCS URI: {model_dir}")
    bucket_name, prefix = m.group(1), m.group(2)

    # SavedModel path prefix
    export_prefix = prefix.rstrip("/") + "/saved_model/"

    client = storage.Client()
    # If any object exists under saved_model/, call it good.
    blobs = list(client.list_blobs(bucket_name, prefix=export_prefix, page_size=10))
    if not blobs:
        raise RuntimeError(f"SavedModel not found at gs://{bucket_name}/{export_prefix}. "
                           f"Ensure your trainer calls:\n"
                           f'  tf.saved_model.save(model, os.path.join(args.model_dir, "saved_model"))')
    return f"SavedModel exists at gs://{bucket_name}/{export_prefix}"


# ---------- Helper to build worker pool specs (TPU by default) ----------

def tpu_worker_pool(
    train_image_uri: str,
    gcs_data: str,
    classes: str,
    model_dir: str,
    tpu_machine_type: str,
    tpu_topology: str,
):
    """
    Single-replica TPU worker pool. Defaults should be quota-friendly (v5e 1 chip).
    Examples:
      - 1 chip:  machineType='ct5lp-hightpu-1t', tpuTopology='1x1'
      - 4 chips: machineType='ct5lp-hightpu-4t', tpuTopology='2x2'
      - 8 chips: machineType='ct5lp-hightpu-8t', tpuTopology='2x4'
    """
    return [{
        "replicaCount": 1,
        "machineSpec": {
            "machineType": tpu_machine_type,
            "tpuTopology": tpu_topology,
        },
        "containerSpec": {
            "imageUri": train_image_uri,
            "command": ["python", "-u", "/trainer/train_furniture_tpu.py"], 
            "args": [
                "--gcs_data", gcs_data,
                "--classes", classes,
                "--epochs", "10",
                "--batch_size", "32",
                "--model_dir", model_dir,
                "--device", "tpu"
            ],
        },
    }]




def cpu_worker_pool(
    train_image_uri: str,
    gcs_data: str,
    classes: str,
    model_dir: str,
    machine_type: str = "e2-standard-8",
    epochs: int = 10,
    initial_epochs: int = 5,
    batch_size: int = 32,
    fine_tune_layers: int = 40, 
    img_size: int=224
):
    """
    Build a CPU-only worker_pool_specs list for Vertex AI CustomTrainingJob.

    Args:
      train_image_uri: Full Artifact Registry URI of your trainer image
      (e.g., us-central1-docker.pkg.dev/<PROJECT>/mlimages/tf-furniture-tpu:latest).
      gcs_data: GCS prefix with your training images (gs://.../furniture/raw).
      classes: Comma-separated label list ("chair,sofa,table,bed").
      model_dir: GCS path where the trainer should write outputs (gs://.../models/...).
      machine_type: Vertex AI compute shape for training (default "e2-standard-8").
      epochs: Small default for a smoke test (adjust as needed).
      batch_size: Mini-batch size for your trainer.

    Returns:
      A list suitable for CustomTrainingJobOp(worker_pool_specs=...).
    """
    return [{
        "replicaCount": 1,
        "machineSpec": {
            "machineType": machine_type,   # e.g., "e2-standard-8", "n1-standard-8", etc.
        },
        "containerSpec": {
            "imageUri": train_image_uri,
            "command": ["python", "-u", "/trainer/train_furniture_tpu.py"], ## ENTRYPOINT in DOCKER SHould do this itself - this overides that.
            "args": [
                "--gcs_data", gcs_data,
                "--classes", classes,
                "--img_size", str(img_size),
                "--batch_size", str(batch_size),
                "--epochs", str(epochs),
                "--initial_epochs", str(initial_epochs),
                "--model_dir", model_dir,
                "--fine_tune_layers 40",
                "--train_split 0.85",
                "--device", "cpu"
            ],
        },
    }]



# ---------- The pipeline ----------

    


#---- 
@dsl.pipeline(name="tpu-train-upload-deploy")
def pipeline(
    project_id: str,
    region: str = "us-central1",
    # Trainer container image you pushed to Artifact Registry
    train_image_uri: str = "",
    # Data & model locations (same bucket/region recommended)
    gcs_data: str = "",                                  # e.g., gs://.../furniture/raw
    classes: str = "almirah,chair,fridge,table,tv",
    model_dir: str = "gs://kubeflow-furniture-poc-mlops-furniture-data/models/tf-furniture",   # e.g., gs://.../models/tf-furniture
    # Model/endpoint display names
    model_display_name: str = "tf-furniture-tpu",
    endpoint_display_name: str = "furniture-tf-endpoint",
    # The service account that should run the training job (email, not numeric ID)
    training_service_account: str = "vertex-pipelines-sa@kubeflow-furniture-poc.iam.gserviceaccount.com",  # e.g., vertex-pipelines-sa@<PROJECT>.iam.gserviceaccount.com
    # TPU sizing (defaults are the smallest v5e slice; adjust if you have quota)
    use_tpu: str = "false",
    tpu_machine_type: str = "ct5lp-hightpu-1t",
    tpu_topology: str = "1x1",
):
    # 0) Preflight: confirm images exist
    preflight = validate_inputs(gcs_data=gcs_data)

    # 1) Parameter based CPU or TPU 
    with dsl.If(use_tpu == "true"):
        train_op = CustomTrainingJobOp(
                display_name="tpu-train-customjob",
                project=project_id,
                location=region,
                worker_pool_specs=tpu_worker_pool(
                    train_image_uri=train_image_uri,
                    gcs_data=gcs_data,
                    classes=classes,
                    model_dir=model_dir,
                    tpu_machine_type=tpu_machine_type,
                    tpu_topology=tpu_topology,
                ),
                base_output_directory=model_dir,
                service_account=training_service_account,  # ensure the runtime SA is used
            ).after(preflight)

    with dsl.Else():
            train_op = CustomTrainingJobOp(
                display_name="cpu-train",
                project=project_id, 
                location=region,
#                worker_pool_specs=cpu_worker_pool(train_image_uri, gcs_data, classes, model_dir),
                worker_pool_specs=cpu_worker_pool(train_image_uri, gcs_data, classes ,model_dir, machine_type="e2-standard-8", epochs=10, initial_epochs=5, batch_size=32, fine_tune_layers= 40, img_size=224),
                base_output_directory=model_dir, service_account=training_service_account,
            )   #).set_caching_options(False)

    # 2) Postflight: verify SavedModel exists at {model_dir}/saved_model/
    postflight = verify_export(model_dir=model_dir).after(train_op)


    # 3) Import an UnmanagedContainerModel artifact that points to the SavedModel
    unmanaged = dsl.importer(
        # artifact_uri=f"{model_dir}/saved_model",  # model_dir should NOT end with '/'
        artifact_uri="gs://kubeflow-furniture-poc-mlops-furniture-data/models/tf-furniture/saved_model",
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                # Match TF version used to train; 2-15 is fine if you trained with 2.15
                "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"
            }
        },
    ).after(postflight)

    #4) Upload the model 
    upload_op = ModelUploadOp(
        project=project_id,
        location=region,
        display_name=model_display_name,
        unmanaged_container_model=unmanaged.outputs["artifact"],
    )

    # 5) Create Endpoint and deploy the model
    endpoint_op = EndpointCreateOp(
        project=project_id,
        location=region,
        display_name=endpoint_display_name,
    ).set_caching_options(False)


    _ = ModelDeployOp(
        endpoint=endpoint_op.outputs["endpoint"],
        model=upload_op.outputs["model"],
        deployed_model_display_name=f"{model_display_name}-deployed",
        traffic_split={"0": 100},
        # Dedicated resources (CPU serving). Use automatic_* instead if you want autoscaling.
        dedicated_resources_machine_type="n1-standard-4",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )




