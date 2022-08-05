---
layout: model
title: Summarization for en language
author: John Snow Labs
name: test_model_hub_upload
date: 2022-08-05
tags: [en, open_source]
task: Summarization
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is ussed for Summarization and this model works with en language

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/test_model_hub_upload_en_4.0.0_3.0_1659701078106.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
restaurant_pipeline = PretrainedPipeline("nerdl_restaurant_100d_pipeline", lang = "en")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|test_model_hub_upload|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|25.9 MB|

## Included Models

- NerDLModel