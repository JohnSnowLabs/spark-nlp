---
layout: model
title: English custom_dataset_deberta_xsmall_pipeline pipeline DeBertaEmbeddings from Sandy1857
author: John Snow Labs
name: custom_dataset_deberta_xsmall_pipeline
date: 2025-01-23
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained DeBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`custom_dataset_deberta_xsmall_pipeline` is a English model originally trained by Sandy1857.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/custom_dataset_deberta_xsmall_pipeline_en_5.5.1_3.0_1737643178470.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/custom_dataset_deberta_xsmall_pipeline_en_5.5.1_3.0_1737643178470.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("custom_dataset_deberta_xsmall_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("custom_dataset_deberta_xsmall_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|custom_dataset_deberta_xsmall_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|265.5 MB|

## References

https://huggingface.co/Sandy1857/custom-dataset-deberta-xsmall

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaEmbeddings