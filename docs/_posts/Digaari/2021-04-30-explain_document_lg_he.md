---
layout: model
title: Explain Document pipeline for Hebrew (explain_document_lg)
author: John Snow Labs
name: explain_document_lg
date: 2021-04-30
tags: [hebrew, ner, he, open_source, explain_document_lg, pipeline]
task: Named Entity Recognition
language: he
edition: Spark NLP 3.0.2
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The explain_document_lg is a pre-trained pipeline that we can use to process text with a simple pipeline that performs basic processing steps and recognizes entities. It performs most of the common text processing tasks on your dataframe

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/explain_document_lg_he_3.0.2_3.0_1619775273050.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/explain_document_lg_he_3.0.2_3.0_1619775273050.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline
pipeline = PretrainedPipeline('explain_document_lg', lang = 'he')
annotations =  pipeline.fullAnnotate(""היי, מעבדות ג'ון סנו!"")[0]
annotations.keys()
```
```scala
val pipeline = new PretrainedPipeline("explain_document_lg", lang = "he")
val result = pipeline.fullAnnotate("היי, מעבדות ג'ון סנו!")(0)
```


{:.nlu-block}
```python
import nlu
nlu.load("he.explain_document").predict("""היי, מעבדות ג'ון סנו!""")
```

</div>

## Results

```bash
+----------------------+------------------------+----------------------+---------------------------+--------------------+---------+
|                  text|                document|              sentence|                      token|                 ner|ner_chunk|
+----------------------+------------------------+----------------------+---------------------------+--------------------+---------+
| היי ג'ון מעבדות שלג! |[ היי ג'ון מעבדות שלג! ]|[היי ג'ון מעבדות שלג!]|[היי, ג'ון, מעבדות, שלג, !]|[O, B-PERS, O, O, O]|   [ג'ון]|
+----------------------+------------------------+----------------------+---------------------------+--------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|explain_document_lg|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.0.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- NerDLModel
- NerConverter