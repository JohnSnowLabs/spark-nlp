---
layout: model
title: Pipeline to Classify Texts into 4 News Categories
author: John Snow Labs
name: bert_sequence_classifier_age_news_pipeline
date: 2022-02-23
tags: [ag_news, news, bert, bert_sequence, classification, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_sequence_classifier_age_news_en](https://nlp.johnsnowlabs.com/2021/11/07/bert_sequence_classifier_age_news_en.html) which is imported from `HuggingFace`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_age_news_pipeline_en_3.4.0_3.0_1645616467835.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
news_pipeline = PretrainedPipeline("bert_sequence_classifier_age_news_pipeline", lang = "en")

news_pipeline.annotate("Microsoft has taken its first step into the metaverse.")
```
```scala
val news_pipeline = new PretrainedPipeline("bert_sequence_classifier_age_news_pipeline", lang = "en")

news_pipeline.annotate("Microsoft has taken its first step into the metaverse.")
```
</div>

## Results

```bash
['Sci/Tech']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_age_news_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|42.4 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification
