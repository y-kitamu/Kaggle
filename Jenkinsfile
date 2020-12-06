pipeline {
    agent { docker { image 'python:latest' } }
    stages {
        stage('build') {
            steps {
                sh '''
                   cd CassavaLeafDisease
                   python setup.py bdist_wheel
                   pip install kaggle
                   KAGGLE_CONFIG_DIR=/projects/Kaggle kaggle datasets version -m "jenkins ci" -p dist/
                   '''
            }
        }
    }
}