---
layout: model
title: My pipeline for news type classification
author: MariiaDemchenko
name: my_nlp_pipeline
date: 2021-10-24
tags: [en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.1
spark_version: 3.0
supported: false
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Some description

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/MariiaDemchenko/my_nlp_pipeline_en_3.3.1_3.0_1635100919098.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
|Model Name:|my_nlp_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.3.1+|
|License:|Open Source|
|Edition:|Community|
|Language:|en|

## Included Models

- DocumentAssembler
- TokenizerModel
- NormalizerModel
- StopWordsCleaner
- LemmatizerModel
- WordEmbeddingsModel
- SentenceEmbeddings
- ClassifierDLModel