---
layout: model
title: French drbert_casm2_pipeline pipeline BertForTokenClassification from medkit
author: John Snow Labs
name: drbert_casm2_pipeline
date: 2024-09-05
tags: [fr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`drbert_casm2_pipeline` is a French model originally trained by medkit.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/drbert_casm2_pipeline_fr_5.5.0_3.0_1725563296489.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/drbert_casm2_pipeline_fr_5.5.0_3.0_1725563296489.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("drbert_casm2_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("drbert_casm2_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drbert_casm2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|408.2 MB|

## References

https://huggingface.co/medkit/DrBERT-CASM2

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification