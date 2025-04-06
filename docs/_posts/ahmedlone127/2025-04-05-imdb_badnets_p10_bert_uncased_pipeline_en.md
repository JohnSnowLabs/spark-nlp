---
layout: model
title: English imdb_badnets_p10_bert_uncased_pipeline pipeline BertForSequenceClassification from dilarayavuz
author: John Snow Labs
name: imdb_badnets_p10_bert_uncased_pipeline
date: 2025-04-05
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`imdb_badnets_p10_bert_uncased_pipeline` is a English model originally trained by dilarayavuz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/imdb_badnets_p10_bert_uncased_pipeline_en_5.5.1_3.0_1743822362431.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/imdb_badnets_p10_bert_uncased_pipeline_en_5.5.1_3.0_1743822362431.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("imdb_badnets_p10_bert_uncased_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("imdb_badnets_p10_bert_uncased_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|imdb_badnets_p10_bert_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/dilarayavuz/imdb-badnets-p10-bert-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification