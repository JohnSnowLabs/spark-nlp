---
layout: model
title: English seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline pipeline ViTForImageClassification from JLB-JLB
author: John Snow Labs
name: seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline
date: 2025-04-05
tags: [en, open_source, pipeline, onnx]
task: Image Classification
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline` is a English model originally trained by JLB-JLB.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline_en_5.5.1_3.0_1743849855704.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline_en_5.5.1_3.0_1743849855704.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|seizure_vit_jlb_231108_indoiranian_languages_adjusted_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.3 MB|

## References

https://huggingface.co/JLB-JLB/seizure_vit_jlb_231108_iir_adjusted

## Included Models

- ImageAssembler
- ViTForImageClassification