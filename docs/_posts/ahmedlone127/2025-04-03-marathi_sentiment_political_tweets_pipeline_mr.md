---
layout: model
title: Marathi marathi_sentiment_political_tweets_pipeline pipeline BertForSequenceClassification from l3cube-pune
author: John Snow Labs
name: marathi_sentiment_political_tweets_pipeline
date: 2025-04-03
tags: [mr, open_source, pipeline, onnx]
task: Text Classification
language: mr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marathi_sentiment_political_tweets_pipeline` is a Marathi model originally trained by l3cube-pune.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marathi_sentiment_political_tweets_pipeline_mr_5.5.1_3.0_1743668105326.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marathi_sentiment_political_tweets_pipeline_mr_5.5.1_3.0_1743668105326.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("marathi_sentiment_political_tweets_pipeline", lang = "mr")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("marathi_sentiment_political_tweets_pipeline", lang = "mr")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marathi_sentiment_political_tweets_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|mr|
|Size:|892.9 MB|

## References

References

https://huggingface.co/l3cube-pune/marathi-sentiment-political-tweets

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification