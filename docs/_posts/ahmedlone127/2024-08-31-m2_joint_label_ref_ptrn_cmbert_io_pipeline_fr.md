---
layout: model
title: French m2_joint_label_ref_ptrn_cmbert_io_pipeline pipeline CamemBertForTokenClassification from nlpso
author: John Snow Labs
name: m2_joint_label_ref_ptrn_cmbert_io_pipeline
date: 2024-08-31
tags: [fr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`m2_joint_label_ref_ptrn_cmbert_io_pipeline` is a French model originally trained by nlpso.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/m2_joint_label_ref_ptrn_cmbert_io_pipeline_fr_5.4.2_3.0_1725139766137.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/m2_joint_label_ref_ptrn_cmbert_io_pipeline_fr_5.4.2_3.0_1725139766137.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("m2_joint_label_ref_ptrn_cmbert_io_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("m2_joint_label_ref_ptrn_cmbert_io_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|m2_joint_label_ref_ptrn_cmbert_io_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|412.9 MB|

## References

https://huggingface.co/nlpso/m2_joint_label_ref_ptrn_cmbert_io

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForTokenClassification