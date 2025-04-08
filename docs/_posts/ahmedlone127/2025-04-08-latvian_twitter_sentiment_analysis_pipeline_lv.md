---
layout: model
title: Latvian latvian_twitter_sentiment_analysis_pipeline pipeline BertForSequenceClassification from matiss
author: John Snow Labs
name: latvian_twitter_sentiment_analysis_pipeline
date: 2025-04-08
tags: [lv, open_source, pipeline, onnx]
task: Text Classification
language: lv
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`latvian_twitter_sentiment_analysis_pipeline` is a Latvian model originally trained by matiss.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/latvian_twitter_sentiment_analysis_pipeline_lv_5.5.1_3.0_1744116820827.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/latvian_twitter_sentiment_analysis_pipeline_lv_5.5.1_3.0_1744116820827.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("latvian_twitter_sentiment_analysis_pipeline", lang = "lv")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("latvian_twitter_sentiment_analysis_pipeline", lang = "lv")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|latvian_twitter_sentiment_analysis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|lv|
|Size:|414.9 MB|

## References

https://huggingface.co/matiss/Latvian-Twitter-Sentiment-Analysis

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification