import requests
import json


class PushToHub:
    @staticmethod
    def upload_to_modelshub_and_fill_form_API(model_data:dict,GIT_TOKEN:str):
        """Pushes model to Hub .

        Keyword Arguments:
        model_data:Dictionary containing info about the model such as Name and Language
        GIT_TOKEN: Token required for pushing to hub."""
        
        r1 = requests.post('https://modelshub.johnsnowlabs.com/api/v1/models', data=json.dumps(model_data), headers={
                'Content-type': 'application/json',
                'Authorization': f'Bearer {GIT_TOKEN}'
            })

        if r1.status_code == 201:
            r2 = requests.post(
                'https://modelshub.johnsnowlabs.com/api/v1/models/%s/file' % r1.json()['id'],
                data=open(model_data['model_zip_path'], 'rb'), headers={
                    'Authorization': f'Bearer {GIT_TOKEN}'
                })
            if r2.status_code == 200:
                print(r2.json()['message'])
        else: 

            print(f"Something Went Wrong During the Upload. Got Status Code: {r1.status_code}")


    @staticmethod
    def create_docs(name:str, task:str, title:str, ):


from python.sparknlp.upload_to_hub import PushToHub
sample_upload = {
    "name":"analyze_sentiment_ml",
    "task":'Sentiment Analysis',
    'title':'Analyze Sentiment Machine Learning ',
    'sparkVersion':"3.0",
    'sparknlpVersion':'Spark NLP 4.0.0',
    'language':'en',
    'license':'Open Source',
    'description':'''The analyze_sentiment_ml is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and predicts sentiment  .
         It performs most of the common text processing tasks on your dataframe''',
    'pythonCode':'''from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("analyze_sentiment_ml", "en")

result = pipeline.annotate("""I love johnsnowlabs!  """)''',
'model_zip_path':'pos_ud_bokmaal_nb_3.4.0_3.0_1641902661339.zip'

}

PushToHub.upload_to_modelshub_and_fill_form_API(sample_upload,GitToken )


