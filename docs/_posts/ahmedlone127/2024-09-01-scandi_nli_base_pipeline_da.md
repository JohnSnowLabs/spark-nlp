---
layout: model
title: Danish scandi_nli_base_pipeline pipeline BertForZeroShotClassification from alexandrainst
author: John Snow Labs
name: scandi_nli_base_pipeline
date: 2024-09-01
tags: [da, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: da
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`scandi_nli_base_pipeline` is a Danish model originally trained by alexandrainst.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/scandi_nli_base_pipeline_da_5.4.2_3.0_1725201672957.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/scandi_nli_base_pipeline_da_5.4.2_3.0_1725201672957.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("scandi_nli_base_pipeline", lang = "da")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("scandi_nli_base_pipeline", lang = "da")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|scandi_nli_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|da|
|Size:|666.2 MB|

## References

https://huggingface.co/alexandrainst/scandi-nli-base

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForZeroShotClassification