# Deploying to AWS – ECR + App Runner Guide

This guide covers how to push the backend Docker image to **Amazon ECR**
and run it on **AWS App Runner** (the simplest serverless option on AWS).

---

## Prerequisites

1. An AWS account (free tier works for initial testing)
2. AWS CLI installed: `pip install awscli`
3. Docker installed and running
4. Configure AWS CLI:
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Key, region (e.g., us-east-1), and output format (json)
   ```

---

## Step 1 – Build the Docker Image Locally

```bash
# From the project root
cd backend

docker build -t churn-backend:1.0.0 .

# Test that it works locally
docker run -p 8000:8000 churn-backend:1.0.0
# Visit http://localhost:8000/health
```

---

## Step 2 – Push to Amazon ECR

ECR (Elastic Container Registry) is AWS's private Docker registry.

```bash
# 1. Set your AWS account ID and region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# 2. Create an ECR repository
aws ecr create-repository --repository-name churn-backend --region $AWS_REGION

# 3. Log in to ECR
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS \
  --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 4. Tag the image
docker tag churn-backend:1.0.0 \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/churn-backend:1.0.0

# 5. Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/churn-backend:1.0.0
```

---

## Step 3 – Deploy on AWS App Runner

App Runner pulls the image from ECR and runs it without you managing any servers.

```bash
# Option A: via AWS Console
# Go to → App Runner → Create service
# Source: Container registry → ECR → select your image
# Port: 8000
# Health check: /health
# Environment variables: MODEL_VERSION=1.0.0 LOG_LEVEL=INFO

# Option B: via CLI (create apprunner.json first)
aws apprunner create-service \
  --service-name churn-backend \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'"$AWS_ACCOUNT_ID"'.dkr.ecr.'"$AWS_REGION"'.amazonaws.com/churn-backend:1.0.0",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000"
      }
    },
    "AutoDeploymentsEnabled": false
  }'
```

After deployment, App Runner gives you a URL like:
`https://abcdefg.us-east-1.awsapprunner.com`

---

## Step 4 – (Alternative) EC2 Deployment

If you prefer a traditional VM:

```bash
# 1. Launch an EC2 t2.micro (free tier) with Amazon Linux 2
# 2. SSH into the instance
ssh -i your-key.pem ec2-user@YOUR_EC2_IP

# 3. Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# 4. Pull the image from ECR (same login command as above)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker pull $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/churn-backend:1.0.0

# 5. Run the container
docker run -d -p 80:8000 \
  -e MODEL_VERSION=1.0.0 \
  $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/churn-backend:1.0.0

# 6. Open port 80 in the EC2 Security Group (inbound rule: HTTP, 0.0.0.0/0)
# 7. Visit http://YOUR_EC2_IP/health
```

---

## AWS-Ready Docker Notes

The Dockerfile in `backend/` is already AWS-ready:
- Uses `python:3.11-slim` (works on any Linux architecture including ARM/Graviton)
- Runs as non-root user (required for some ECS/App Runner configurations)
- `HEALTHCHECK` is defined (used by App Runner)
- `PORT` env var is respected (App Runner sets its own port)
- No secrets baked in (all via environment variables)

---

## Cost Estimate (Free Tier)

| Service | Free Tier | After Free Tier |
|---------|-----------|-----------------|
| ECR | 500 MB storage free | $0.10/GB/month |
| App Runner | 2M requests/month free | $0.064/vCPU-hour |
| EC2 t2.micro | 750 hours/month free (1 year) | ~$8/month |
