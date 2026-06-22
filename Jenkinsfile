pipeline {
    agent any

    environment {
        AWS_REGION = 'us-east-1'
        ECR_REPO = '049559537457.dkr.ecr.us-east-1.amazonaws.com/adidas-sales-predictor'
        IMAGE_TAG = "v${BUILD_NUMBER}"
        EKS_CLUSTER = 'adidas-ml-cluster'
        MLFLOW_URI = 'http://44.211.229.174:5000'
    }

    stages{
        stage('Checkout'){
            steps{
                echo "PUlling code from GIT"
                checkout scm
            }
        }

        stage('Python Setup'){
            steps{
                echo "Installing the python packages"
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --quiet pandas numpy scikit-learn xgboost joblib pyyaml mlflow boto3 s3fs
                 '''
            }
        }

        stage('Data Ingesstion'){
            steps{
                echo "Runing Data ingestion"
                sh '''
                    . venv/bin/activate
                    python src/data_ingestion.py
                '''
            }
        }

        stage('Data Validation'){
            steps{
                echo "Runing Data Validation"
                sh '''
                    . venv/bin/activate
                    python src/data_validation.py
                '''                
            }
        }

        stage('Feature Engineering'){
            steps{
                echo "Runing Feature Engineering"
                sh '''
                    . venv/bin/activate
                    python src/feature_engineering.py
                '''                
            }
        }

        stage('Model Traning'){
            steps{
                echo "Runing Model Traning"
                sh '''
                    . venv/bin/activate
                    python src/train.py
                '''                   
            }
        }

        stage('Evaluate Model'){
            steps{
                echo "Runing Model Evaluate"
                sh '''
                    . venv/bin/activate
                    python src/evaluate.py
                '''                   
            }
        }
        
        stage('Register Model'){
            steps{
                echo "Runing Model Register"
                sh '''
                    . venv/bin/activate
                    python src/register_model.py
                '''                   
            }
        }                

        stage('Docker Build'){
            steps{
                echo 'Building Docker image...'
                sh '''
                    docker build -t ${ECR_REPO}:${IMAGE_TAG} .
                    docker tag ${ECR_REPO}:${IMAGE_TAG} ${ECR_REPO}:latest
                '''                   
            }
        } 

         stage('Push to ECR'){
            steps{
                echo 'Pushing image to ECR...'
                sh '''
                    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO}
                    docker push ${ECR_REPO}:${IMAGE_TAG}
                    docker push ${ECR_REPO}:latest
                '''                   
            }
        }

          stage('Deploy to EKS'){
            steps{
                echo 'Deploying to EKS...'
                sh '''
                    kubectl set image deployment/sales-predictor model-api=${ECR_REPO}:${IMAGE_TAG}
                    kubectl rollout status deployment/sales-predictor --timeout=120s
                '''                   
            }
        } 
         

    }
    post{
        success{
            echo "pieline completed"
        }
        failure{
            echo "pipeline failed"
        }
        always{
            sh 'docker system prune -f || true'
        }
    }



}