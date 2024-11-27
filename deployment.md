# Deployment steps

are simply defined in Makefile

## Pre-requisite:

You created an .azure file

```bash
#.azure
SUBSCRIPTION="Your-subscription-name"
RESOURCE_GROUP="Your-resource-group"
LOCATION="Your-resource-group-location"
ENVIRONMENT="Your-container-environment"
API_NAME="Your-container-app-name"
API_VERSION="v1.0.0"
IMAGE_NAME="Your-image-name"
ACR_NAME="Your-acr-name"
MINREPLICAS=1
MAXREPLICAS=5
ALLOCATED_CPU="0.5"
ALLOCATED_MEMORY=1.0Gi
PROJECT_ID="Your-project-id-tag"
OWNERSERVICE="Your-owner-service-tag"
PIC="Your-pic-tag"
```
## Update 

This explains how to update the (deployed) container app after making some changes to the code:

```bash
make create-update
```

This will build the Docker image, push to the ACR, and finally update the container app.

