---
layout: model
title: Multilingual sentiment_multilingual_distilbert_pipeline pipeline DistilBertForSequenceClassification from Mukalingam0813
author: John Snow Labs
name: sentiment_multilingual_distilbert_pipeline
date: 2024-09-19
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentiment_multilingual_distilbert_pipeline` is a Multilingual model originally trained by Mukalingam0813.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_multilingual_distilbert_pipeline_xx_5.5.0_3.0_1726741391537.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentiment_multilingual_distilbert_pipeline_xx_5.5.0_3.0_1726741391537.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentiment_multilingual_distilbert_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentiment_multilingual_distilbert_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentiment_multilingual_distilbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|507.4 MB|

## References

https://huggingface.co/Mukalingam0813/sentiment-multilingual-distilbert

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification