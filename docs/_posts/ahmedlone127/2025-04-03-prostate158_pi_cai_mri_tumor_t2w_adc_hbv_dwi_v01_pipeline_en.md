---
layout: model
title: English prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline pipeline ViTForImageClassification from Emilio407
author: John Snow Labs
name: prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline
date: 2025-04-03
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

Pretrained ViTForImageClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline` is a English model originally trained by Emilio407.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline_en_5.5.1_3.0_1743724726015.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline_en_5.5.1_3.0_1743724726015.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|prostate158_pi_cai_mri_tumor_t2w_adc_hbv_dwi_v01_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.1 GB|

## References

https://huggingface.co/Emilio407/Prostate158-PI-CAI-MRI-Tumor-T2W-ADC-HBV-DWI-v01

## Included Models

- ImageAssembler
- ViTForImageClassification