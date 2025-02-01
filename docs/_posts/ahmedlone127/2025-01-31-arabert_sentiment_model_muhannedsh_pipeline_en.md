---
layout: model
title: English arabert_sentiment_model_muhannedsh_pipeline pipeline BertForSequenceClassification from Muhannedbsh
author: John Snow Labs
name: arabert_sentiment_model_muhannedsh_pipeline
date: 2025-01-31
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arabert_sentiment_model_muhannedsh_pipeline` is a English model originally trained by Muhannedbsh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arabert_sentiment_model_muhannedsh_pipeline_en_5.5.1_3.0_1738327143021.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arabert_sentiment_model_muhannedsh_pipeline_en_5.5.1_3.0_1738327143021.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arabert_sentiment_model_muhannedsh_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arabert_sentiment_model_muhannedsh_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arabert_sentiment_model_muhannedsh_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|507.3 MB|

## References

https://huggingface.co/Muhannedbsh/arabert-sentiment-model-MuhannedSh

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification