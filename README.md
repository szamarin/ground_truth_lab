# SageMaker Ground Truth Hands-On Lab

## Overview
In this Hands-On Lab you will gain an overview of of how SageMaker Ground Truth can faciliate an end-to-end data labeling and model training workflow where you will
1. Setup a Ground Turth labeling job to annotate pairs of headshot photos as being of the same or different person
2. Incorporate the output of the labeling job into your data preprocessing workflow
3. Train a facial recognition model with TensorFlow
4. Deploy the model as a REST API Endpoint


## Getting Started
1. Login to your AWS Account
2. Launch the CloudFormation Stack
3. Navigate to SageMaker in the AWS Management Console and ope JupyterLab on your notebook Instance
[<img src="notebook_images/cloudformation-launch-stack.png">](https://console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/new?stackName=ground-truth-lab&templateURL=https://s3.amazonaws.com/mldayasset.corp.amazon.com/ml_immersion_day_cloud_formation.yaml)
4. Follow instructions in the *ground_truth_lab.ipynb* notebook