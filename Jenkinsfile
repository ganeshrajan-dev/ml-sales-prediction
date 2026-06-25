pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        ECR_REPO = '049559537457.dkr.ecr.us-east-1.amazonaws.com/adidas-sales-predictor'
        IMAGE_TAG = "v${BUILD_NUMBER}"
        EKS_CLUSTER = 'adidas-ml-cluster'
        MLFLOW_URI = 'http://100.31.134.18:5000'

    }

    stages{
        stage('Checkout'){
            steps{
                echo "Pulling code from Source Branch: ${BRANCH_NAME}"
                checkout scm
            }                        
        }
        stage('PR Raised - Validation Started'){
            when{
                changeRequest()
            }
            steps{
                echo "=========================================="
                echo "PR #${CHANGE_ID} has been raised!"
                echo "Source Branch: ${CHANGE_BRANCH}"
                echo "Target Branch: ${CHANGE_TARGET}"
                echo "PR Title: ${CHANGE_TITLE}"
                echo "PR Author: ${CHANGE_AUTHOR}"
                echo "Running validation pipeline..."
                echo "=========================================="

            }
        }
        stage('PythonStep'){
            steps{
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --quiet pandas numpy scikit-learn xgboost joblib pyyaml mlflow boto3 s3fs sagemaker
                '''
            }
        }
        stage('Data Ingestion'){
            steps{
                sh '''
                    . venv/bin/activate
                    python src/data_ingestion.py
                '''
            }
        }
        stage('Data Validation'){
            steps{
                sh '''
                    . venv/bin/activate
                    python src/data_validation.py
                '''
            }
        }
        stage('Feature Engineering'){
            steps{
                sh '''
                    . venv/bin/activate
                    python src/feature_engineering.py
                '''
            }
        }
        stage('Model Training'){
            steps{
                sh '''
                    . venv/bin/activate
                    python src/train.py
                '''
            }
        }
        stage('Evaluate Model'){
            stpes{
                sh '''
                    . venv/bin/activate
                    python src/evaluate.py
                '''
            }
        }
        stage('Register Model'){
            when{
                anyof {
                    branch 'dev'
                    branch 'main'
                }
            }
            steps{
                sh '''
                    . venv/bin/activate
                    python src/register_model.py
                '''
            }
        }
        stage('Docker Build'){
            when{
                anyof{
                    branch 'dev'
                    branch 'main'
                }
            }
            steps{
                sh '''
                    docker build -t ${ECR_REPO}:${IMAGE_TAG} .
                    docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_REPO}:latest
                '''
            }
        }
        stage('Push to ECR'){
            when{
                anyof{
                    branch 'dev'
                    branch 'main'
                }
            }
            steps{
                sh '''
                    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}
                    docker push ${ECR_REPO}:${IMAGE_TAG}
                    docker push ${ECR_REPO}:latest
                '''
            }
        }
        stage('Deploy to EKS'){
            when{
                anyof{
                    branch 'dev'
                    branch 'main'
                }
            }
            steps{
                sh '''
                    aws eks update-kubeconfig --name ${EKS_CLUSTER} --region ${AWS_REGION}
                    kubectl set image deployment/sales-predictor model-api=${ECR_REPO}:${IMAGE_TAG}
                    kubectl rollout status deployment/sales-predictor --timeout=120s
                '''
            }
        }
    }
    post{
        success{
            echo "Pipeline completed for branch: ${BRANCH_NAME}"
        }
        failure{
            echo"Pipeline failed for branch: ${BRANCH_NAME}"
        }
        always{
            sh 'docker system prune -f || true'
        }
    }
}
