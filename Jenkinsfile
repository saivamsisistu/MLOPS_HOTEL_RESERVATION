pipeline{
    agent any
    environment{
        VENV_DIR='venv'
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
                pip isntall --upgrade pip 
                pip install -e .
                '''
            }
        }
    }
     
}