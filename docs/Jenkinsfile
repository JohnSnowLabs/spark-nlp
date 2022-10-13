#!groovy

pipeline {
    agent any
    stages {
        stage('Create Search Index Publish - Incremental'){
            when {
                branch 'master'
            }
            environment {
                GIT_REPO_URL = "https://git:${GITHUB_TOKEN}@github.com/${SPARK_NLP_REPO}.git"
            }
            steps {
                script {
                    env.GIT_AUTHOR_NAME = sh(script: 'git --no-pager show -s --format="%an" $GIT_COMMIT', returnStdout: true).trim()
                    env.GIT_AUTHOR_EMAIL = sh(script: 'git --no-pager show -s --format="%ae" $GIT_COMMIT', returnStdout: true).trim()
                    dir("docs") {
                        sh 'docker-compose build'
                        echo "Starting incremental build ..."
                        def returnStatus = sh (
                                script: 'docker-compose run -e ELASTICSEARCH_URL=${ELASTICSEARCH_URL} -e ELASTICSEARCH_ACCESS_TOKEN=${ELASTICSEARCH_ACCESS_TOKEN} --rm spark_nlp_build_incremental',
                                returnStatus: true)
                        if (returnStatus == 11) {
                            echo "New editions encountered. Starting forced full build..."
                            sh 'docker-compose run -e ELASTICSEARCH_URL=${ELASTICSEARCH_URL} -e ELASTICSEARCH_ACCESS_TOKEN=${ELASTICSEARCH_ACCESS_TOKEN} --rm spark_nlp_build_forced'
                        }
                    }
                }
            }
        }
    }
}
