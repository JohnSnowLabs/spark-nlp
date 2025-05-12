---
layout: model
title: English abte_restaurants_sentiment_distilbert_pipeline pipeline DistilBertForSequenceClassification from Khoivudang1209
author: John Snow Labs
name: abte_restaurants_sentiment_distilbert_pipeline
date: 2025-04-01
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`abte_restaurants_sentiment_distilbert_pipeline` is a English model originally trained by Khoivudang1209.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/abte_restaurants_sentiment_distilbert_pipeline_en_5.5.1_3.0_1743526311969.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/abte_restaurants_sentiment_distilbert_pipeline_en_5.5.1_3.0_1743526311969.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("abte_restaurants_sentiment_distilbert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("abte_restaurants_sentiment_distilbert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|abte_restaurants_sentiment_distilbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/Khoivudang1209/abte-restaurants-sentiment-distilbert

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification