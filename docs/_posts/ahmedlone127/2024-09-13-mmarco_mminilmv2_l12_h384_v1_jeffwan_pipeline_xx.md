---
layout: model
title: Multilingual mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline pipeline XlmRoBertaForSequenceClassification from jeffwan
author: John Snow Labs
name: mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline
date: 2024-09-13
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline` is a Multilingual model originally trained by jeffwan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline_xx_5.5.0_3.0_1726258207064.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline_xx_5.5.0_3.0_1726258207064.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mmarco_mminilmv2_l12_h384_v1_jeffwan_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|399.6 MB|

## References

https://huggingface.co/jeffwan/mmarco-mMiniLMv2-L12-H384-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification