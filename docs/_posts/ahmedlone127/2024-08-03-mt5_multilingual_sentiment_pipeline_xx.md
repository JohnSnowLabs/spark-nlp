---
layout: model
title: Multilingual mt5_multilingual_sentiment_pipeline pipeline T5Transformer from Chirayu
author: John Snow Labs
name: mt5_multilingual_sentiment_pipeline
date: 2024-08-03
tags: [xx, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: xx
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_multilingual_sentiment_pipeline` is a Multilingual model originally trained by Chirayu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_multilingual_sentiment_pipeline_xx_5.4.2_3.0_1722643212751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_multilingual_sentiment_pipeline_xx_5.4.2_3.0_1722643212751.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_multilingual_sentiment_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_multilingual_sentiment_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_multilingual_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.3 GB|

## References

https://huggingface.co/Chirayu/mt5-multilingual-sentiment

## Included Models

- DocumentAssembler
- T5Transformer