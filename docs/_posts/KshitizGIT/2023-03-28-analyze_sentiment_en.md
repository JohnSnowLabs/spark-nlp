---
layout: model
title: test
author: John Snow Labs
name: analyze_sentiment
date: 2023-03-28
tags: [en, licensed]
task: Assertion Status
language: en
edition: Healthcare NLP 3.0.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Now is coming

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/analyze_sentiment_en_3.0.1_3.0_1679980956468.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/analyze_sentiment_en_3.0.1_3.0_1679980956468.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
Some python code
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|analyze_sentiment|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.0.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|5.2 MB|
|Dependencies:|winter|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- NorvigSweetingModel
- ViveknSentimentModel