---
layout: docs
comment: no
header: true
title: API Integration
permalink: /docs/en/alab/api
key: docs-training
modify_date: "2021-09-22"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

All features offered by the Annotation Lab via UI are also accessible via API. The complete API documentation is available on the SWAGGER page of the Annotation Lab. This can be accessed via UI by clicking on the documentation icon on the left lower side of the screen as shown in the image below:

<img class="image image--xl" src="/assets/images/annotation_lab/2.1.0/API access.png" style="width:60%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

<img class="image image--xl" src="/assets/images/annotation_lab/2.1.0/swagger.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

Concrete query examples are provided for each available endpoint. 

<img class="image image--xl" src="/assets/images/annotation_lab/2.1.0/list_of_projects.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>



## Example of creating a new project via API

### Get Client Secret

Get CLIENT_ID and CLIENT_SECRET by following the steps illustrated in the video.

<div class="cell cell--12 cell--lg-6 cell--sm-12"><div class="video-item">{%- include extensions/youtube.html id='5dIh8xwW0UY' -%}<div class="video-descr">Annotation Lab: Collect the Client Secret </div></div></div>

### Call API endpoint

For creating a new project via API you can use the following python script. 


```python
import requests
import json

# URL to Annotation Lab
API_URL = "https://123.45.67.89"
# Add user credentials
USERNAME = "user"
PASSWORD = "password"
# The above video shows how to get CLIENT_ID and CLIENT_SECRET
CLIENT_ID = "..."
CLIENT_SECRET = "..."

PROJECT_NAME = "sample_project"


IDENTITY_MANAGEMENT_URL = API_URL + "/auth/"
IDENTITY_MANAGEMENT_REALM = "master"
HEADERS = {
    "Host": API_URL.replace("http://", "").replace("https://", ""),
    "Origin": API_URL,
    "Content-Type": "application/json",
}


def get_cookies():
    url = f"{IDENTITY_MANAGEMENT_URL}realms/{IDENTITY_MANAGEMENT_REALM}/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    auth_info = requests.post(url, data=data).json()
    cookies = {
        "access_token": f"Bearer {auth_info['access_token']}",
        "refresh_token": auth_info["refresh_token"],
    }
    return cookies


def create_project():
    # GET THIS FROM SWAGGER DOC
    url = f"{API_URL}/api/projects/create"
    data = {
        "project_name": PROJECT_NAME,
        "project_description": "",
        "project_sampling": "uniform",
        "project_instruction": "",
    }
    r = requests.post(
        url, headers=HEADERS, data=json.dumps(data), cookies=get_cookies()
    )
    return r

create_project()

```


