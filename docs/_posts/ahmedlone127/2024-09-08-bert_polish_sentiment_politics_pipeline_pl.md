---
layout: model
title: Polish bert_polish_sentiment_politics_pipeline pipeline BertForSequenceClassification from eevvgg
author: John Snow Labs
name: bert_polish_sentiment_politics_pipeline
date: 2024-09-08
tags: [pl, open_source, pipeline, onnx]
task: Text Classification
language: pl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_polish_sentiment_politics_pipeline` is a Polish model originally trained by eevvgg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_polish_sentiment_politics_pipeline_pl_5.5.0_3.0_1725838948305.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_polish_sentiment_politics_pipeline_pl_5.5.0_3.0_1725838948305.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_polish_sentiment_politics_pipeline", lang = "pl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_polish_sentiment_politics_pipeline", lang = "pl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_polish_sentiment_politics_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pl|
|Size:|495.8 MB|

## References

https://huggingface.co/eevvgg/bert-polish-sentiment-politics

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification