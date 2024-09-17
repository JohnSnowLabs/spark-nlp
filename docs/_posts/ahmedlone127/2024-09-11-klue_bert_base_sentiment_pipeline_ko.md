---
layout: model
title: Korean klue_bert_base_sentiment_pipeline pipeline BertForSequenceClassification from hun3359
author: John Snow Labs
name: klue_bert_base_sentiment_pipeline
date: 2024-09-11
tags: [ko, open_source, pipeline, onnx]
task: Text Classification
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`klue_bert_base_sentiment_pipeline` is a Korean model originally trained by hun3359.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/klue_bert_base_sentiment_pipeline_ko_5.5.0_3.0_1726059564558.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/klue_bert_base_sentiment_pipeline_ko_5.5.0_3.0_1726059564558.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("klue_bert_base_sentiment_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("klue_bert_base_sentiment_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|klue_bert_base_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|414.8 MB|

## References

https://huggingface.co/hun3359/klue-bert-base-sentiment

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification