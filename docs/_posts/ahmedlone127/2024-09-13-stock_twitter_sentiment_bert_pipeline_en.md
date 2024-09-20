---
layout: model
title: English stock_twitter_sentiment_bert_pipeline pipeline DistilBertForSequenceClassification from NazmusAshrafi
author: John Snow Labs
name: stock_twitter_sentiment_bert_pipeline
date: 2024-09-13
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`stock_twitter_sentiment_bert_pipeline` is a English model originally trained by NazmusAshrafi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stock_twitter_sentiment_bert_pipeline_en_5.5.0_3.0_1726262365809.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/stock_twitter_sentiment_bert_pipeline_en_5.5.0_3.0_1726262365809.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("stock_twitter_sentiment_bert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("stock_twitter_sentiment_bert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stock_twitter_sentiment_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/NazmusAshrafi/stock_twitter_sentiment_Bert

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification