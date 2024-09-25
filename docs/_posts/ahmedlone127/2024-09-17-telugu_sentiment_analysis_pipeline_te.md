---
layout: model
title: Telugu telugu_sentiment_analysis_pipeline pipeline AlbertForSequenceClassification from aashish-249
author: John Snow Labs
name: telugu_sentiment_analysis_pipeline
date: 2024-09-17
tags: [te, open_source, pipeline, onnx]
task: Text Classification
language: te
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`telugu_sentiment_analysis_pipeline` is a Telugu model originally trained by aashish-249.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/telugu_sentiment_analysis_pipeline_te_5.5.0_3.0_1726605971118.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/telugu_sentiment_analysis_pipeline_te_5.5.0_3.0_1726605971118.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("telugu_sentiment_analysis_pipeline", lang = "te")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("telugu_sentiment_analysis_pipeline", lang = "te")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|telugu_sentiment_analysis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|te|
|Size:|125.9 MB|

## References

https://huggingface.co/aashish-249/Telugu-sentiment_analysis

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification