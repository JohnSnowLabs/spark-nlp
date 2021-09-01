---
layout: docs
header: true
title: Installation
permalink: /docs/en/nlp_server
key: docs-nlp-server
modify_date: "2021-04-17"
---

NLP-Server empowers any programming language to use over 4000+ industry grade NLP Models in 300+ Languages with just a simple REST API call.
Each of these models  peform at state-of-the-art accuracy and can leverage CPU and GPU for maximum speed.
These models are refered to as so called `spells`, provided by the [NLU library](https://nlu.johnsnowlabs.com/) and 
powered by the [historically most accurate](TODO)
and most widely used NLP library in the industry, [Spark NLP](http://nlp.johnsnowlabs.com/).


You can setup NLP-Server as a Docker Machine in any enviroment or get it via the `AWS Marketplace` in just 1 click.


## AWS-Marketplace Setup
todo

## Docker Setup

```shell
docker pull <TODO UPLOAD IMAGE SO WE CAN PULL>
docker run --rm -it -p 5000:5000 spark-nlp-server:latest

```


## Simple usage examples

Python:
```python
import requests
# Invoke Processing with tokenization spell
r = requests.post(f'http://localhost:9000/api/results',json={"spell": "tokenize","data": "I love NLU! <3"})
# Use the uuid to get your processed data
uuid = r.json()['uuid']


# Get status of processing
r = requests.get(f'http://localhost:9000/api/results/{uuid}').json()
>>> {'sentence': {'0': ['I love NLU! <3']}, 'document': {'0': 'I love NLU! <3'}, 'token': {'0': ['I', 'love', 'NLU', '!', '<3']}}

# Get status of processing
r = requests.get(f'http://localhost:9000/api/results/{uuid}/status').json
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



## Full API Docs
See [this page](TODO) for the full API docs