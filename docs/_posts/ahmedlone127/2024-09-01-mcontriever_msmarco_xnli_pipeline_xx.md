---
layout: model
title: Multilingual mcontriever_msmarco_xnli_pipeline pipeline BertForZeroShotClassification from mjwong
author: John Snow Labs
name: mcontriever_msmarco_xnli_pipeline
date: 2024-09-01
tags: [xx, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: xx
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mcontriever_msmarco_xnli_pipeline` is a Multilingual model originally trained by mjwong.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mcontriever_msmarco_xnli_pipeline_xx_5.4.2_3.0_1725202432638.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mcontriever_msmarco_xnli_pipeline_xx_5.4.2_3.0_1725202432638.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mcontriever_msmarco_xnli_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mcontriever_msmarco_xnli_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mcontriever_msmarco_xnli_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|665.8 MB|

## References

https://huggingface.co/mjwong/mcontriever-msmarco-xnli

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForZeroShotClassification