---
layout: model
title: English f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline pipeline RoBertaForSequenceClassification from IDQO
author: John Snow Labs
name: f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline
date: 2025-02-04
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline` is a English model originally trained by IDQO.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline_en_5.5.1_3.0_1738691134814.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline_en_5.5.1_3.0_1738691134814.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|f146e71c_342e_4d0f_a28b_edd4f18efab5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|438.0 MB|

## References

https://huggingface.co/IDQO/f146e71c-342e-4d0f-a28b-edd4f18efab5

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification