pipeline {
    agent { docker { image 'python:latest' } }
    stages {
        stage('build') {
            steps {
                sh 'cd CassavaLeafDisease && python setup.py bdist_wheel'
            }
        }
    }
}