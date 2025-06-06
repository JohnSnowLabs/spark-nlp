---
layout: model
title: Portuguese bert_large_portuguese_cased_hatebr_pipeline pipeline BertForSequenceClassification from ruanchaves
author: John Snow Labs
name: bert_large_portuguese_cased_hatebr_pipeline
date: 2024-09-13
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
language: pt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_large_portuguese_cased_hatebr_pipeline` is a Portuguese model originally trained by ruanchaves.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_portuguese_cased_hatebr_pipeline_pt_5.5.0_3.0_1726202219760.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_large_portuguese_cased_hatebr_pipeline_pt_5.5.0_3.0_1726202219760.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_large_portuguese_cased_hatebr_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_large_portuguese_cased_hatebr_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_large_portuguese_cased_hatebr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|1.3 GB|

## References

https://huggingface.co/ruanchaves/bert-large-portuguese-cased-hatebr

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification