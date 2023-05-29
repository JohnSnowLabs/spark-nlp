---
layout: model
title: Entity Recognition Pipeline (Large, Spanish)
author: John Snow Labs
name: entity_recognizer_lg
date: 2023-05-21
tags: [es, open_source]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The entity_recognizer_lg is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and recognizes entities.
It performs most of the common text processing tasks on your dataframe

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_es_4.4.2_3.0_1684630228186.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_es_4.4.2_3.0_1684630228186.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("entity_recognizer_lg", "es")

result = pipeline.annotate("""I love johnsnowlabs! """)
```



{:.nlu-block}
```python
import nlu
nlu.load("es.ner.lg").predict("""I love johnsnowlabs! """)
```

</div>

{:.model-param}

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("entity_recognizer_lg", "es")

result = pipeline.annotate("""I love johnsnowlabs! """)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.ner.lg").predict("""I love johnsnowlabs! """)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|entity_recognizer_lg|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|2.5 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter