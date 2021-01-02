pipeline {
    agent { dockerfile {
          filename 'Dockerfile'
          dir 'docker'
          } }
    stages {
        stage('build') {
            steps {
                sh '''
                   cd CassavaLeafDisease
                   python setup.py bdist_wheel
                   rm -rf dist/*
                   cp /projects/Kaggle/CassavaLeafDisease/ext/* dist/
                   cp /projects/Kaggle/CassavaLeafDisease/dist/dataset-metadata.json dist/
                   KAGGLE_CONFIG_DIR=/projects/Kaggle kaggle datasets version -m "jenkins ci" -p dist
                   cp dist/*.whl /projects/Kaggle/CassavaLeafDisease/dist/
                   '''
            }
        }
    }
}