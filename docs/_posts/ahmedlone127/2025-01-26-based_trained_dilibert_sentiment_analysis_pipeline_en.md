---
layout: model
title: English based_trained_dilibert_sentiment_analysis_pipeline pipeline DistilBertForSequenceClassification from dmo0798
author: John Snow Labs
name: based_trained_dilibert_sentiment_analysis_pipeline
date: 2025-01-26
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`based_trained_dilibert_sentiment_analysis_pipeline` is a English model originally trained by dmo0798.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/based_trained_dilibert_sentiment_analysis_pipeline_en_5.5.1_3.0_1737905150506.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/based_trained_dilibert_sentiment_analysis_pipeline_en_5.5.1_3.0_1737905150506.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("based_trained_dilibert_sentiment_analysis_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("based_trained_dilibert_sentiment_analysis_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|based_trained_dilibert_sentiment_analysis_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/dmo0798/based_trained_dilibert_sentiment_analysis

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification