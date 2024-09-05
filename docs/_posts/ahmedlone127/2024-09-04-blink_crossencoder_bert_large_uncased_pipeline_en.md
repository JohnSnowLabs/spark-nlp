---
layout: model
title: English blink_crossencoder_bert_large_uncased_pipeline pipeline BertForSequenceClassification from shomez
author: John Snow Labs
name: blink_crossencoder_bert_large_uncased_pipeline
date: 2024-09-04
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`blink_crossencoder_bert_large_uncased_pipeline` is a English model originally trained by shomez.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/blink_crossencoder_bert_large_uncased_pipeline_en_5.5.0_3.0_1725432935906.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/blink_crossencoder_bert_large_uncased_pipeline_en_5.5.0_3.0_1725432935906.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("blink_crossencoder_bert_large_uncased_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("blink_crossencoder_bert_large_uncased_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|blink_crossencoder_bert_large_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/shomez/blink-crossencoder-bert-large-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification