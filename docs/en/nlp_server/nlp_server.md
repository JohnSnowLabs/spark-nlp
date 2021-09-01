---
layout: docs
header: true
title: NLP Server
permalink: /docs/en/nlp_server
key: docs-nlp-server
modify_date: "2021-04-17"
---


This is a ready to use NLP Server for analyzing text documents using NLU library. Over 4000+ industry grade NLP Models in 300+ Languages  are available to use via a simple and intuitive UI, without writing a line of code. For more expert users and more complex tasks, NLP Server also provides a REST API that can be used to process high amounts of data.

The models, refered to as `spells`, are provided by the [NLU library](https://nlu.johnsnowlabs.com/) and 
powered by the most widely used NLP library in the industry, [Spark NLP](http://nlp.johnsnowlabs.com/). 

NLP Server is `free` for everyone to download and use. There is no limitation in the amout of text to analyze. 

You can setup NLP-Server as a Docker Machine in any enviroment or get it via the `AWS Marketplace` in just 1 click.

## AWS-Marketplace Setup
todo

## Docker Setup

```shell

docker run -p 5000:5000 johnsnowlabs/nlp-server:latest

```
## Web UI
The Web UI is accessible at the followin URL: http://localhost:5000/
It allows a very simple and intuitive interaction with the NLP Server. 
As a first step the user chooses the spell from the first dropdown. All NLU spells are available. 
Then the user has to provide a text document for analysis. This can be done by either copy/pasting text on the text box, or by uploading a csv/json file. 
After selecting the grouping option, the user clicks on the `Preview` button to get the results for the first 10 rows of text. 

<img class="image image--xl" src="/assets/images/nlp_server/web_UI.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>

## REST API

NLP Server includes a REST API which can be used to process any amount of data using NLU. Once you deploy the NLP Server, you can access the API documentation at the following URL [http://0.0.0.0:5000/docs](http://0.0.0.0:5000/docs).


<img class="image image--xl" src="/assets/images/nlp_server/api_docs.png" style="width:100%; align:center; box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);"/>


## How to use in Python 

```python
import requests
# Invoke Processing with tokenization spell
r = requests.post(f'http://localhost:5000/api/results',json={"spell": "tokenize","data": "I love NLU! <3"})
# Use the uuid to get your processed data
uuid = r.json()['uuid']


# Get status of processing
r = requests.get(f'http://localhost:5000/api/results/{uuid}').json()
>>> {'sentence': {'0': ['I love NLU! <3']}, 'document': {'0': 'I love NLU! <3'}, 'token': {'0': ['I', 'love', 'NLU', '!', '<3']}}

# Get status of processing
r = requests.get(f'http://localhost:5000/api/results/{uuid}/status').json
>>> {'status': {'code': 'success', 'message': None}}

```

[comment]: <> (Javascript:)
[comment]: <> (```javascript)
[comment]: <> (todo)
[comment]: <> (```)

[comment]: <> (## Reccomended Machine Size)
[comment]: <> (The reccomended machine size is 32GB with at least 4 cores.)

[comment]: <> (NLP Server offeres a variety of NLP models that can range from 6MB to 6GB.)
[comment]: <> (For your particular use-case, you might not use some of the large spells and can thus use a smaller VM to run the server.)
[comment]: <> (|Avaiable Spells | RAM  |CPU  |AWS Sample Machine| Azure Sample Machine | GCP Sample Machine|)
[comment]: <> (|---------------|-----|-----|------------------|----------------------|----------------------|)
[comment]: <> (| 5             | 5GB |  6 | abc1              | 123b                 | asd                  |)

[comment]: <> (## Full VM Spell  Compatiblity Overview)

[comment]: <> (TODO)

[comment]: <> (## VM Spell Benchmarks )

[comment]: <> (TODO )

