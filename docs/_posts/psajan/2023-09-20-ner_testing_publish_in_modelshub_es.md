---
layout: model
title: ner_testing_publish_in_modelshub
author: John Snow Labs
name: ner_testing_publish_in_modelshub
date: 2023-09-20
tags: [es, open_source, tensorflow]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.3.1
spark_version: 3.2
supported: true
engine: tensorflow
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Testing model publishing

## Predicted Entities

`Medicine`, `MedicalCondition`, `Pathogen`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_testing_publish_in_modelshub_es_4.3.1_3.2_1695221488615.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_testing_publish_in_modelshub_es_4.3.1_3.2_1695221488615.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Test test
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_testing_publish_in_modelshub|
|Type:|ner|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|2.5 MB|
|Dependencies:|glove_100d|