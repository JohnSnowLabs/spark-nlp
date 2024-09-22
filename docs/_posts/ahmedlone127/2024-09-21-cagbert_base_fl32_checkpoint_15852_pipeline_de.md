---
layout: model
title: German cagbert_base_fl32_checkpoint_15852_pipeline pipeline BertForTokenClassification from MSey
author: John Snow Labs
name: cagbert_base_fl32_checkpoint_15852_pipeline
date: 2024-09-21
tags: [de, open_source, pipeline, onnx]
task: Named Entity Recognition
language: de
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cagbert_base_fl32_checkpoint_15852_pipeline` is a German model originally trained by MSey.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cagbert_base_fl32_checkpoint_15852_pipeline_de_5.5.0_3.0_1726890061211.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cagbert_base_fl32_checkpoint_15852_pipeline_de_5.5.0_3.0_1726890061211.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cagbert_base_fl32_checkpoint_15852_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cagbert_base_fl32_checkpoint_15852_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cagbert_base_fl32_checkpoint_15852_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|409.8 MB|

## References

https://huggingface.co/MSey/CaGBERT-base_fl32_checkpoint-15852

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification