---
layout: model
title: Entity Recognizer LG
author: John Snow Labs
name: entity_recognizer_lg
date: 2022-06-25
tags: [pt, open_source]
task: Named Entity Recognition
language: pt
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The entity_recognizer_lg is a pretrained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and recognizes entities.
It performs most of the common text processing tasks on your dataframe

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_pt_4.0.0_3.0_1656130333737.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/entity_recognizer_lg_pt_4.0.0_3.0_1656130333737.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline("entity_recognizer_lg", "pt")

result = pipeline.annotate("""I love johnsnowlabs! """)
```



{:.nlu-block}
```python
import nlu
nlu.load("pt.ner.lg").predict("""I love johnsnowlabs! """)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|entity_recognizer_lg|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|2.5 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter