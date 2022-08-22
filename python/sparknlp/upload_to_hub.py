import requests
import json
from typing import List
import sparknlp
import os
import zipfile


class PushToHub:
    list_of_tasks = [  # List  of available tasks in Modelhub
        "Named Entity Recognition",
        'Text Classification',
        'Text Generation',
        'Sentiment Analysis',
        'Translation',
        'Question Answering',
        'Summarization',
        'Sentence Detection',
        'Embeddings',
        'Language Detection',
        'Stop Words Removal',
        'Word Segmentation',
        'Part of Speech Tagging',
        'Lemmatization',
        'Chunk Mapping',
        'Spell Check',
        'Dependency Parser',
        'Pipeline Public']

    def zip_directory(folder_path: str, zip_path: str):
        """Zips folder for pushing to hub.

        folder_path:Path to the folder to zip.
        zip_path:Path of the zip file to create."""

        with zipfile.ZipFile(zip_path, mode='w') as zipf:
            len_dir_path = len(folder_path)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path[len_dir_path:])

    def unzip_directory(zip_path: str):
        """Unzips Model to check for required files for upload.

        Keyword Arguments:
        zip_path:Zip Path to unzip.
        """

    def check_for_required_info(model_data: dict):
        """Checks if the required fields exist in given dictionary  and fills any remaining fields.

        Keyword Arguments: 
        model_data: The model data to check.
        """
        
        list_of_required_fields = ['name', 'task', 'language', 'pythonCode', 'model_zip_path']

        if model_data['task'] not in PushToHub.list_of_tasks:
            list_of_tasks_string_version = "\n".join(PushToHub.list_of_tasks)
            raise ValueError(
                f"""Invalid task, please pick one of the following tasks\n{list_of_tasks_string_version}""")

        if model_data['model_zip_path'].endswith(".zip"):
            with zipfile.ZipFile(model_data['model_zip_path']) as modelfile:
                if 'metadata/part-00000' not in modelfile.namelist():
                    raise ValueError("The Model is not a Spark Saved Model.")
        else:
            if not os.path.exists(f"{model_data['model_zip_path']}/metadata/part-00000"):
                raise ValueError("The Model is not a Spark Saved Model.")

    def push_to_hub(name: str,
                    language: str,
                    model_zip_path: str,
                    task: str,
                    pythonCode: str,
                    GIT_TOKEN: str,
                    title: str = None,
                    tags: List[str] = None,
                    dependencies: str = None,
                    description: str = None,
                    predictedEntities: str = None,
                    sparknlpVersion: str = None,
                    howToUse: str = None,
                    liveDemo: str = None,
                    runInColab: str = None,
                    scalaCode: str = None,
                    nluCode: str = None,
                    results: str = None,
                    dataSource: str = None,
                    includedModels: str = None,
                    benchmarking: str = None,
                    ) -> str:
        """Pushes model to Hub.

        Keyword Arguments:
        model_data:Dictionary containing info about the model such as Name and Language.
        GIT_TOKEN: Token required for pushing to hub.
        """

        model_data = {item: value for (item, value) in locals().items() if value is not None}
        PushToHub.check_for_required_info(model_data)
        model_data = PushToHub.create_docs(model_data)

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
                return r2.json()['message']
        else:
            print(f"Something Went Wrong During the Upload. Got Status Code: {r1.status_code}")
            return f"Something Went Wrong During the Upload. Got Status Code: {r1.status_code}"

    def create_docs(dicionary_for_upload: dict) -> dict:
        """Adds fields in the dictionary for pushing to hub.

        Keyword Arguments:
        dictionary_for_upload: The dictionary to add keys to.
        """

        dicionary_for_upload['sparkVersion'] = "3.0"
        dicionary_for_upload['license'] = 'Open Source'
        dicionary_for_upload['supported'] = False

        if 'sparknlpVersion' not in dicionary_for_upload.keys():
            dicionary_for_upload['sparknlpVersion'] = "Spark NLP " + sparknlp.version()

        if 'description' not in dicionary_for_upload.keys():
            dicionary_for_upload[
                'description'] = f"This model is used for {dicionary_for_upload['task']} and this model works with {dicionary_for_upload['language']} language"

        if 'title' not in dicionary_for_upload.keys():
            dicionary_for_upload[
                'title'] = f"{dicionary_for_upload['task']} for {dicionary_for_upload['language']} language"

        if os.path.isdir(dicionary_for_upload['model_zip_path']):
            PushToHub.zip_directory(dicionary_for_upload['model_zip_path'],
                                    f"{dicionary_for_upload['model_zip_path']}.zip")
            dicionary_for_upload['model_zip_path'] = dicionary_for_upload['model_zip_path'] + '.zip'
        return dicionary_for_upload
