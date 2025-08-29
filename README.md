
# Furniture MLOps PoC  
## Vertex AI AutoML + Custom TensorFlow (MobileNetV2 Transfer Learning) + Kubeflow Pipelines ##


1. **Vertex AI AutoML Image** (no-code) — trained and hosted by Vertex AI.
2. **Custom TensorFlow model** (MobileNetV2 transfer learning) — trained by a **Kubeflow (KFP v2)** pipeline on Vertex AI and deployed to a Vertex AI **Endpoint**.


## Getting started with gcloud - all the basic stuff ##
```bash
git --version
pip --version
gcloud --version
gcloud components update
gcloud auth login
gcloud services enable aiplatform.googleapis.com cloudbuild.googleapis.com compute.googleapis.com storage googleapis.com
```

<br>
<br>

## Create the Cloud Storage Bucket for the Project & Region ##
```bash
export PROJECT_ID=kubeflow-furniture-poc
export REGION=us-central1
export BUCKET=gs://${PROJECT_ID}-mlops-furniture-data
## make bucket 
gsutil mb -l $REGION $BUCKET
#details of bucket
gsutil ls -L -b $BUCKET
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud auth application-default login
```
<BR>
<BR>


## DATA PREP ##
### Prepare the image data locally after downloading it from Kaggle

Kaggle Furniture Dataset => https://www.kaggle.com/datasets/udaysankarmukherjee/furniture-image-dataset
<BR>
Folder structure => 
<BR>
data/
	almirah/
	chair/
	fridge/
	table/
	tv/
<BR>
<BR>

## Upload this data to Google Cloud Storage  (GCS) ##
```bash
gsutil -m rsync -r data $BUCKET/data/

# verify  
gsutil du gs://kubeflow-furniture-poc-mlops-furniture-data | wc -l
gsutil du gs://kubeflow-furniture-poc-mlops-furniture-data/data | wc -l
gsutil du gs://kubeflow-furniture-poc-mlops-furniture-data/data/almirah | wc -l
gsutil du gs://kubeflow-furniture-poc-mlops-furniture-data/data/chair | wc -l
gsutil du gs://kubeflow-furniture-poc-mlops-furniture-data/data/fridge | wc -l
gsutil du gs://kubeflow-furniture-poc-mlops-furniture-data/data/table | wc -l
gsutil du gs://kubeflow-furniture-poc-mlops-furniture-data/data/tv | wc -l
```

<BR>
<BR>

## Upload the label mapping csv to the bucket ##
```bash
gsutil cp image_data.csv $BUCKET/data/metadata/image_data.csv
```

<BR>
<BR>
<BR>
<BR>
<BR>
<BR>








## FOR CUSTOM MODEL ##
### Create an Artifact Registry repo and build the trainer image ###
```bash
gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create mlimages --repository-format=Docker --location=$REGION
gcloud auth configure-docker $REGION-docker.pkg.dev
export TRAIN_IMG=$REGION-docker.pkg.dev/$PROJECT_ID/mlimages/tf-furniture-tpu:latest
docker build -f Dockerfile.tpu -t $TRAIN_IMG .
docker push $TRAIN_IMG
```

<BR>
<BR>
<BR>


## Create the local python env  ##
```bash 
python -m venv env  
source env/scripts/activate
pip install -r requirements.txt
python.exe -m pip install --upgrade pip
```


## See which project your gcloud CLI is set to ##
```bash
gcloud config list project
# Enable required APIs (add this one explicitly)
gcloud services enable cloudresourcemanager.googleapis.com aiplatform.googleapis.com storage.googleapis.com compute.googleapis.com
```

<BR>
<BR>
<BR>

## If there are errors, fix issue with Service Accounts - set up correctly ##
```bash
export PROJECT_ID=kubeflow-furniture-poc
gcloud iam service-accounts create vertex-pipelines-sa   --display-name="Vertex Pipelines SA" --project=$PROJECT_ID

# Let the SA use Vertex AI
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-pipelines-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Let the SA read/write your pipeline bucket (artifacts and model_dir)
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-mlops-furniture-data \
  --member="serviceAccount:vertex-pipelines-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Let the SA pull the trainer image from Artifact Registry (reader on the project or repo)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-pipelines-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"


#Allow your user to run as that service account (actAs / “Service Account User”):

gcloud iam service-accounts add-iam-policy-binding \
  vertex-pipelines-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --member="user:YOUR_GOOGLE_ACCOUNT_EMAIL" \
  --role="roles/iam.serviceAccountUser"

```

<BR>
<BR>
<BR>

## ERROR SCENARIO ## 
The Vertex AI Error indicates pipeline’s custom training job needs to pull your training container image from Artifact Registry. The error says the Vertex AI Service Agent: service-481983728986@gcp-sa-aiplatform-cc.iam.gserviceaccount.com does not have permission to read from your repo: 

```bash
projects/kubeflow-furniture-poc/locations/us-central1/repositories/mlimages
PROJECT_ID="kubeflow-furniture-poc"
REGION="us-central1"
REPO="mlimages"
PROJECT_NUMBER="$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')"
```
<BR>
<BR>
<BR>

## ERROR SCENARIO ## 
Granting Artifact Registry Reader lets Vertex AI control-plane agents pull images for jobs/components. See Artifact Registry access control and common guidance to add roles/artifactregistry.reader for principals that must pull images.

```bash
gcloud artifacts repositories add-iam-policy-binding "$REPO" \
  --location="$REGION" --project="$PROJECT_ID" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

# standard Vertex AI service agent
gcloud artifacts repositories add-iam-policy-binding "$REPO" \
  --location="$REGION" --project="$PROJECT_ID" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

gcloud artifacts repositories add-iam-policy-binding "$REPO" \
  --location="$REGION" --project="$PROJECT_ID" \
  --member="serviceAccount:vertex-pipelines-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

# If you’re using the Compute Engine default SA instead:
gcloud artifacts repositories add-iam-policy-binding "$REPO" \
  --location="$REGION" --project="$PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"


# Make sure all required APIS are enabled
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com


# Show who has access on the repo
gcloud artifacts repositories get-iam-policy "$REPO" --location="$REGION" --project="$PROJECT_ID"

# Verify the image exists at the expected hostname (region-specific)
gcloud artifacts docker images list "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO" --include-tags

```

<BR>
<BR>
<BR>

## ERROR SCENARIO ## 
You may see “Listed 0 items” because no image was ever pushed to your Artifact Registry repo (or it was pushed to the wrong hostname/region/repo). Get a Linux-compatible training image built and pushed to us-central1-docker.pkg.dev/kubeflow-furniture-poc/mlimages/tf-furniture-tpu:latest

```bash

# Show active project
gcloud config list project

# List your Artifact Registry repos (confirm "mlimages" exists and region)
gcloud artifacts repositories list

# If "mlimages" is missing, create it in us-central1:
gcloud artifacts repositories create mlimages  --location=us-central1 --repository-format=docker

# Authenticate Docker to Artifact Registry
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

# Manually Build the Training Image and push it from windows PowerShell
$PROJECT_ID="kubeflow-furniture-poc"
$REGION="us-central1"
$REPO="mlimages"
$IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/tf-furniture-tpu:latest"

# Optional: pull the public base image first (helps on first build)
docker pull us-docker.pkg.dev/vertex-ai/training/tf-tpu-pod-base-cp310:latest

# Build (force Linux image on Windows)
docker build --platform linux/amd64 -f Dockerfile.tpu -t $IMAGE_URI .

# Push
docker push $IMAGE_URI

#In Git Bash /Command line
PROJECT_ID=kubeflow-furniture-poc
REGION=us-central1
REPO=mlimages
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/tf-furniture-tpu:latest"
docker pull us-docker.pkg.dev/vertex-ai/training/tf-tpu-pod-base-cp310:latest
docker build --platform linux/amd64 -f Dockerfile.tpu -t "$IMAGE_URI" .
docker push "$IMAGE_URI"

#VERIFY THAT THE IMAGE exists
gcloud artifacts docker images list   "us-central1-docker.pkg.dev/$PROJECT_ID/$REPO" --include-tags

#POINT THE PIPELINE TO THIS EXACT IMAGE
TRAIN_IMG = f"us-central1-docker.pkg.dev/{PROJECT_ID}/mlimages/tf-furniture-tpu:latest"


#If Job Fails to determine what happened
# list recent custom jobs
gcloud ai custom-jobs list --region us-central1 --limit 5

# pick the JOB_ID of the failing one and stream its logs:
gcloud ai custom-jobs stream-logs JOB_ID --region us-central1


#ENSURE ONCE MORE 

# Vertex AI user (create jobs, models, endpoints)
gcloud projects add-iam-policy-binding $PROJECT_ID `
  --member="serviceAccount:$SA" `
  --role="roles/aiplatform.user"

# Bucket read/write (pipeline_root & model_dir bucket)
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-mlops-furniture-data `
  --member="serviceAccount:$SA" `
  --role="roles/storage.objectAdmin"


# Artifact Registry reader (to pull train image)
gcloud artifacts repositories add-iam-policy-binding mlimages  --location=$REGION --project=$PROJECT_ID   --member="serviceAccount:$SA"   --role="roles/artifactregistry.reader"



# Allow you to impersonate that SA (Token Creator + User)
gcloud iam service-accounts add-iam-policy-binding $SA   --member="user:shyam.vai@gmail.com"   --role="roles/iam.serviceAccountTokenCreator"   --project=$PROJECT_ID

# Check the Service Account SA can actually read/write:
# write a temp file via the SA
echo test > t.txt
gcloud storage cp t.txt gs://$PROJECT_ID-mlops-furniture-data/perm-test.txt  --impersonate-service-account=$SA
gcloud storage cat "gs://kubeflow-furniture-poc-mlops-furniture-data/perm-test.txt" --impersonate-service-account=$SA


# list images via the SA
gcloud artifacts docker images list  "us-central1-docker.pkg.dev/$PROJECT_ID/mlimages"  --include-tags --impersonate-service-account=$SA

# If that works, you’ve proven your user can impersonate the SA, and the SA can write/read your bucket.
#  LIST SA LIST
gcloud iam service-accounts list --project kubeflow-furniture-poc --format="table(email,uniqueId)"
```

<BR>
<BR>
<BR>

## KEEP GETTING ERRORS DUE TO NO TPU QUOTA  ##
Do a sanity check to see what quota is actually there . Quick quota sanity check from CLI.
```bash
gcloud beta quotas info list  --project=$PROJECT_ID  --service=aiplatform.googleapis.com  --filter="metric:custom_model_training_tpu_v5e"
```
<BR>
<BR>
<BR>


## FOR NOW SWITCH TO CPU  and later on come back to TPU
SEE IF THE MODEL GOT SAVED 
```bash
gcloud storage ls "$MODEL_DIR/saved_model/**"

#LOOKS LIKE A SERVICE ACCOUNT ISSUE .. it did not have model registry access 
gcloud projects add-iam-policy-binding $PROJECT_ID   --member="serviceAccount:$SA"   --role="roles/aiplatform.user" 

#ONLY IF THIS STILL FAILS - ADMIN access
gcloud projects add-iam-policy-binding $PROJECT_ID  --member="serviceAccount:$SA"  --role="roles/aiplatform.admin"

# Full object access to the bucket used by PIPELINE_ROOT and MODEL_DIR
gcloud storage buckets add-iam-policy-binding "$BUCKET"  --member="serviceAccount:$SA"  --role="roles/storage.objectAdmin"

# Artifact Registry Access 
gcloud artifacts repositories add-iam-policy-binding $AR_REPO   --location=$REGION --project=$PROJECT_ID --member="serviceAccount:$SA" --role="roles/artifactregistry.reader"

# get Project number
gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)"


# Even when your pipeline SA uploads the model, deployment uses Google-managed service agents to read the SavedModel from GCS. 
# Give them viewer on your bucket:
export SA_PRED="service-481983728986@gcp-sa-aiplatform.iam.gserviceaccount.com"
export SA_CC="service-481983728986@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "$BUCKET"  --member="serviceAccount:$SA_PRED" --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding "$BUCKET"  --member="serviceAccount:$SA_CC"   --role="roles/storage.objectViewer"
```

<BR>
<BR>
<BR>
<BR>

DEPLOYED info
predict_custom_trained_model_sample(
    project="481983728986",
    endpoint_id="482064380523970560",
    location="us-central1",
    instances={ "instance_key_1": "value", ...}
)

## ERROR - sometimes the training python does not get picket up properly. That means the TRAINER IMAGE is not right. ##
Update Docker File again for Transfer Learning changes  and build docker file again
```bash
REGION=us-central1
PROJECT_ID=kubeflow-furniture-poc
REPO=mlimages
TAG=mobilenetv2-v2-20250827   # <-- new tag

# Your Dockerfile should COPY the trainer into the image, e.g. /trainer/train_furniture_tpu.py
gcloud builds submit   --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/trainer:$TAG"   --project "$PROJECT_ID"

#Build & push to Artifact Registry
#Replace the variables with your values and use a new tag every time you change code:

REGION=us-central1
PROJECT_ID=kubeflow-furniture-poc
REPO=mlimages
TAG=mobilenetv2-v2-20250827   # pick a fresh tag

gcloud builds submit  --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/trainer:$TAG"  --project "$PROJECT_ID"

# Sanity-check the image locally
# Ensures your container really has the new flags the pipeline will pass.
docker run --rm $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/trainer:$TAG --help


# When in doubt - ensure auth
gcloud auth application-default login

# If model deployment failed then check the list of endpoints. 
gcloud ai endpoints list --project=kubeflow-furniture-poc --region=us-central1
```
