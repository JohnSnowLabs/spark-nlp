---
layout: model
title: Russian rubert_tiny_nli_terra_v0_pipeline pipeline BertForSequenceClassification from Marwolaeth
author: John Snow Labs
name: rubert_tiny_nli_terra_v0_pipeline
date: 2025-01-25
tags: [ru, open_source, pipeline, onnx]
task: Text Classification
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rubert_tiny_nli_terra_v0_pipeline` is a Russian model originally trained by Marwolaeth.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rubert_tiny_nli_terra_v0_pipeline_ru_5.5.1_3.0_1737801287916.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rubert_tiny_nli_terra_v0_pipeline_ru_5.5.1_3.0_1737801287916.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rubert_tiny_nli_terra_v0_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rubert_tiny_nli_terra_v0_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rubert_tiny_nli_terra_v0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|69.6 MB|

## References

https://huggingface.co/Marwolaeth/rubert-tiny-nli-terra-v0

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification