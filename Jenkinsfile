pipeline{
    agent any
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                echo 'Cloning Github to Jenkins........'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/saivamsisistu/MLOPS_HOTEL_RESERVATION.git']])

            }
        }
    }
}