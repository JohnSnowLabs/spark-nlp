---
layout: model
title: Multilingual english_bert_base_multilingual_uncased_sentiment_run2_pipeline pipeline BertForSequenceClassification from gunkaynar
author: John Snow Labs
name: english_bert_base_multilingual_uncased_sentiment_run2_pipeline
date: 2024-09-26
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`english_bert_base_multilingual_uncased_sentiment_run2_pipeline` is a Multilingual model originally trained by gunkaynar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/english_bert_base_multilingual_uncased_sentiment_run2_pipeline_xx_5.5.0_3.0_1727309277059.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/english_bert_base_multilingual_uncased_sentiment_run2_pipeline_xx_5.5.0_3.0_1727309277059.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("english_bert_base_multilingual_uncased_sentiment_run2_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("english_bert_base_multilingual_uncased_sentiment_run2_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|english_bert_base_multilingual_uncased_sentiment_run2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|627.8 MB|

## References

https://huggingface.co/gunkaynar/en-bert-base-multilingual-uncased-sentiment_run2

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification