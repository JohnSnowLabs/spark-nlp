import glob
import json
import os
import re

import boto3
from botocore.exceptions import ClientError

directory = os.path.dirname(os.path.dirname(__file__))
directory = directory + os.sep + "_posts"
os.chdir(directory)

files = glob.glob("**/*.md", recursive=True)
session =boto3.Session(profile_name="PROFILE_PASSED_MFA", region_name='us-east-1')
client = session.client('s3')

for matching_file in files:
    try:
        with open(matching_file) as f:
            content = f.read()
        if "[Copy S3 URI]" in content:
            regex = r"(?:\[Copy S3 URI\])\((?P<link>.*?)\){(.*)}"
            matches = re.search(regex, content, re.MULTILINE)
            groups = matches.groups()
            url = groups[0]
            parts = url.replace('s3://', '')
            bucket = parts.split('/')[0]
            key = url.replace(f"s3://{bucket}/", '')
            try: 
                response = client.head_object(Bucket=bucket, Key = key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"File {matching_file} has missing s3 url 's3://{bucket}/{key}'. Removing the model")
                    os.remove(matching_file)
                else:
                    raise e
            except Exception as e:
                print(f"Exception on {matching_file}: {str(e)}")

        else:
            print(f"Copy S3 URI not in {matching_file}")
            
    except Exception as e:
        print(str(e) + " in " + matching_file)
