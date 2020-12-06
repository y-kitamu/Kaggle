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
                   KAGGLE_CONFIG_DIR=/projects/Kaggle kaggle datasets version -m "jenkins ci" -p dist/
                   cp dist/*.whl /projects/kaggle/CassavaLeafDisease/dist/
                   '''
            }
        }
    }
}