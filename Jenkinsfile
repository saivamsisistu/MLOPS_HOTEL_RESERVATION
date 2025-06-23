pipeline{
    agent any
    environment{
        VENV_DIR='venv'
        GCP_PROJECT= "aqueous-ray-462911-u1"
        GCLOUD_PATH= "/var/jenkins_home/google-cloud-sdk/bin"
    }
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                echo 'Cloning Github to Jenkins........'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/saivamsisistu/MLOPS_HOTEL_RESERVATION.git']])

            }
        }
        stage('Setting up of virtual enivironment and installing dependencies'){
            steps{
                echo 'Setting up of virtual enivironment and installing dependencies.....'
                sh '''
                python -m venv ${VENV_DIR}
                . ${VENV_DIR}/bin/activate
                pip install --upgrade pip 
                pip install -e .
                '''
            }
        }
        stage('Building and pushing docker image to GCR '){
            steps{
                withCrendentials([file(credentialsId : 'gcp-key', variable : 'GOOGLE_APPLICATION_CREDENTIALS')])
                script{
                    echo 'Building and pushing docker image to GCR .....'
                    sh '''
                    export PATH=$PATH:$(GCLOUD_PATH)

                    gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                    gcloud config set project ${GCP_PROJECT}

                    gcloud auth configure-docker --quiet

                    docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

                    docker push gcr.io/${GCP_PROJECT}/ml-project:latest 
                    '''
                }
                
            }
        }
    }
     
}