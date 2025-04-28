#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1



echo "Logging in to ECR..."
 aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 565393027942.dkr.ecr.ap-southeast-2.amazonaws.com

echo "Pulling Docker image..."
docker push 565393027942.dkr.ecr.ap-southeast-2.amazonaws.com/spotify-hybrid-recommender-system:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=spotify-hybrid-recommender-system)"]; then
    echo "Stopping existing container..."
    docker stop swiggy-food-delivery-time-prediction
fi

if [ "$(docker ps -aq -f name=spotify-hybrid-recommender-system)"];then
    echo "Removing existing container..."
    docker rm spotify-hybrid-recommender-system
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name spotify-hybrid-recommender-system \
 

echo "Container started successfully."
