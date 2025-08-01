---
layout: model
title: English mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline pipeline XlmRoBertaForSequenceClassification from nreimers
author: John Snow Labs
name: mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline
date: 2025-05-20
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline` is a English model originally trained by nreimers.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline_en_5.5.1_3.0_1747744839143.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline_en_5.5.1_3.0_1747744839143.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mmarco_mminilmv2_l12_h384_v1_nreimers_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|399.6 MB|

## References

References

https://huggingface.co/nreimers/mmarco-mMiniLMv2-L12-H384-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification