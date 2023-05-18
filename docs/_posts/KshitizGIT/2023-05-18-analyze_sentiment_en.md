---
layout: model
title: ksh_test
author: John Snow Labs
name: analyze_sentiment
date: 2023-05-18
tags: [en, open_source]
task: Chunk Mapping
language: en
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Some decription

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/analyze_sentiment_en_3.2.0_3.0_1684412682776.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/analyze_sentiment_en_3.2.0_3.0_1684412682776.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
TEST python
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|analyze_sentiment|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|5.2 MB|
|Dependencies:|dep_1|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- NorvigSweetingModel
- ViveknSentimentModel