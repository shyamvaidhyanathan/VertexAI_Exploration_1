############################  NEW  ##########################

# run_pipeline.py
# ------------------------------------------------------------
# Submit the "tpu-train-upload-deploy" pipeline to Vertex AI.
# Works with KFP v2 + google-cloud-pipeline-components v2.

import os
from kfp import compiler
from google.cloud import aiplatform as aip
from pipeline_tpu_train_deploy import pipeline


# ========= 1) Project / region / service account =========
# TIP: You can set these via environment variables if you prefer.
PROJECT_ID  = os.environ.get("PROJECT_ID",  "kubeflow-furniture-poc")
REGION      = os.environ.get("REGION",      "us-central1")

# IMPORTANT: This must be a *service account email*, not a numeric id.
# Make sure this SA exists and has:
#   - roles/aiplatform.user   (on the project)
#   - roles/storage.objectAdmin (on the bucket you use below)
#   - roles/artifactregistry.reader (on your Artifact Registry repo)
TRAINING_SA = os.environ.get(
    "TRAINING_SA",
    f"vertex-pipelines-sa@{PROJECT_ID}.iam.gserviceaccount.com",
)

# ========= 2) Storage layout (use ONE bucket consistently) =========
# Create this bucket beforehand (regional: us-central1), or the SDK will create
# a staging bucket for you. Keeping everything in one bucket is simplest.
BUCKET_BASE     = os.environ.get("BUCKET_BASE", f"gs://{PROJECT_ID}-mlops-furniture-data")
PIPELINE_ROOT   = os.environ.get("PIPELINE_ROOT", f"{BUCKET_BASE}/pipeline_root")
GCS_DATA        = os.environ.get("GCS_DATA",      f"{BUCKET_BASE}/data") 
MODEL_DIR       = os.environ.get("MODEL_DIR",     f"{BUCKET_BASE}/models/tf-furniture")

# ========= 3) Trainer image in Artifact Registry =========
# Build & push this image beforehand:
#   docker build --platform linux/amd64 -f Dockerfile.tpu \
#     -t us-central1-docker.pkg.dev/<PROJECT_ID>/mlimages/tf-furniture-tpu:latest .
# #   docker push us-central1-docker.pkg.dev/<PROJECT_ID>/mlimages/tf-furniture-tpu:latest
# TRAIN_IMG = os.environ.get(
#     "TRAIN_IMG",
#     f"us-central1-docker.pkg.dev/{PROJECT_ID}/mlimages/tf-furniture-tpu:latest",
# )


TRAIN_IMG = os.environ.get(
    "TRAIN_IMG",
    f"us-central1-docker.pkg.dev/{PROJECT_ID}/mlimages/trainer:mobilenetv2-v2-20250827_vers1.1",
)




# ========= 4) Labels / endpoint display names =========
CLASSES               = os.environ.get("CLASSES", "almirah,chair,fridge,table,tv")  
MODEL_DISPLAY_NAME    = os.environ.get("MODEL_DISPLAY_NAME",    "tf-furniture-tpu")
ENDPOINT_DISPLAY_NAME = os.environ.get("ENDPOINT_DISPLAY_NAME", "furniture-tf-endpoint")

# ========= 5) TPU sizing (keep it small unless you have quota) =========
# If you have 0 TPU quota right now, your run will fail with RESOURCE_EXHAUSTED.
# To get an end-to-end green run today, temporarily switch the pipeline to a CPU
# worker pool (edit the pipeline file) or request TPU v5e quota and re-run.
TPU_MACHINE_TYPE = os.environ.get("TPU_MACHINE_TYPE", "ct5lp-hightpu-1t")
TPU_TOPOLOGY     = os.environ.get("TPU_TOPOLOGY",     "1x1")
USE_TPU          = os.environ.get("USE_TPU","false")

def main() -> None:
    print("=== Config ===")
    print("PROJECT_ID:", PROJECT_ID)
    print("REGION:", REGION)
    print("TRAINING_SA:", TRAINING_SA)
    print("PIPELINE_ROOT:", PIPELINE_ROOT)
    print("GCS_DATA:", GCS_DATA)
    print("MODEL_DIR:", MODEL_DIR)
    print("TRAIN_IMG:", TRAIN_IMG)
    print("TPU_MACHINE_TYPE:", TPU_MACHINE_TYPE)
    print("TPU_TOPOLOGY:", TPU_TOPOLOGY)
    print("USE TPU  ", USE_TPU)
    print("================\n")

    # 1) Compile the pipeline to JSON
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.json",
    )
    print("Compiled pipeline.json")

    # 2) Initialize Vertex AI SDK context
    #    (staging_bucket is optional; it’s fine to omit or to reuse BUCKET_BASE)
    aip.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_BASE)
    print("Initialized Vertex AI SDK")

    # 3) Create and run the pipeline job
    job = aip.PipelineJob(
        display_name="tpu-train-upload-deploy",
        template_path="pipeline.json",
        pipeline_root=PIPELINE_ROOT,   # NOTE: this is NOT a pipeline parameter
        parameter_values={
            # Must match the @dsl.pipeline() signature in pipeline_tpu_train_deploy.py
            "project_id": PROJECT_ID,
            "region": REGION,
            "train_image_uri": TRAIN_IMG,
            "gcs_data": GCS_DATA,
            "classes": CLASSES,
            "model_dir": MODEL_DIR,
            "model_display_name": MODEL_DISPLAY_NAME,
            "endpoint_display_name": ENDPOINT_DISPLAY_NAME,
            "training_service_account": TRAINING_SA,  # ensures the training job runs as this SA
            "tpu_machine_type": TPU_MACHINE_TYPE,
            "tpu_topology": TPU_TOPOLOGY,
            "use_tpu": USE_TPU,
        },
    )

    print("Submitting pipeline job...")
    # Ensure the *submission* also uses the same SA (good practice)
    job.run(service_account=TRAINING_SA, sync=True)
    print("Pipeline submitted. Check Vertex AI > Pipelines for run status.")


if __name__ == "__main__":
    main()
