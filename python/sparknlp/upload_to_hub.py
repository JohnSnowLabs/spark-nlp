import requests
import json


class PushToHub:
    @staticmethod
    def upload_to_modelshub_and_fill_form_API(model_data,GIT_TOKEN):
        
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


