# Jenkinsfile Deploy Step Documentation
#### Last Updated: October 16, 2024


## Overview

This section of the Jenkinsfile is designed to manage the deployment of a Kubernetes job. It includes checks to ensure that all relevant pods are in a "Completed" state before initiating a new deployment. The following steps will outline the key processes involved.

## Step Breakdown

### 1. Waiting for Existing Pods to Complete

```bash
    # Set the timeout duration (15 minutes)
    TIMEOUT=900  # 15 minutes

    # Use timeout to limit the execution duration
    timeout "$TIMEOUT" bash -c '
        while kubectl get pods --no-headers -n default | grep '^central-' | grep 'Running'; do
            echo "Waiting for existing pod to complete..."
            sleep 5
        done
    ' || echo "Timeout reached after 15 minutes."
```

#### Explanation:

- **Set Timeout Duration**: The `TIMEOUT` variable is set to 900 seconds (15 minutes) to limit how long the script will wait for existing pods to complete.
- **Check for Running Pods**: The `timeout` command runs a loop that checks for any pods with names starting with `central-` that are currently in the "Running" state.
  - **`kubectl get pods --no-headers -n default`**: Lists all pods in the `default` namespace without headers.
  - **`grep '^central-'`**: Filters the output to only include pods starting with "central-".
  - **`grep 'Running'`**: Further filters to include only those that are currently running.
- **Waiting Logic**: If such pods are found, it prints a message and sleeps for 5 seconds before checking again.
- **Timeout Handling**: If the loop exceeds the set timeout duration, it will exit, printing "Timeout reached after 15 minutes."

### 2. Deploying a New Job if All Pods are Completed

```bash
    if [ -z "$(kubectl get pods --no-headers -n default | grep '^central-' | awk '{print $3}' | grep -v 'Completed')" ]; then
        echo "All existing central pods are in completed state."
        echo 'Deploying a new one....' 
        aws eks update-kubeconfig --region us-east-1 --name bookzurma-prod-ZeZLUM7s --role-arn arn:aws:iam::745222113226:role/bookzurman-eks-admin-role
        kubectl create -f ioiJob.yml -n default
    else
        echo "Not all existing central pods are completed."
        exit 1
    fi
```

#### Explanation:

- **Check Pod Status**: This step checks if there are any central pods that are not in the "Completed" state.
  - **`[ -z "$( ... )" ]`**: This condition checks if the command inside returns an empty string, meaning there are no non-completed pods.
  - **`awk '{print $3}'`**: Extracts the status column of the pod list.
  - **`grep -v 'Completed'`**: Filters out pods that are in the "Completed" state.
- **Successful Condition**:
  - If all pods are completed, it echoes a message and proceeds to deploy a new job.
  - **AWS EKS Configuration**: Updates the Kubernetes configuration using AWS CLI to ensure the correct context is set for the cluster.
  - **Job Creation**: Creates a new Kubernetes job using the `kubectl create` command with the configuration specified in `ioiJob.yml`.
- **Failure Condition**:
  - If not all pods are completed, it echoes an error message and exits with status `1`, signaling a failure in the pipeline.

