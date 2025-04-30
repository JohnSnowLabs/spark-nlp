---
layout: model
title: Portuguese syas1_ptbr_pipeline pipeline DistilBertForSequenceClassification from 1Arhat
author: John Snow Labs
name: syas1_ptbr_pipeline
date: 2025-03-30
tags: [pt, open_source, pipeline, onnx]
task: Text Classification
language: pt
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`syas1_ptbr_pipeline` is a Portuguese model originally trained by 1Arhat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/syas1_ptbr_pipeline_pt_5.5.1_3.0_1743303809684.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/syas1_ptbr_pipeline_pt_5.5.1_3.0_1743303809684.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("syas1_ptbr_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("syas1_ptbr_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|syas1_ptbr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|249.5 MB|

## References

https://huggingface.co/1Arhat/SYAS1-PTBR

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification