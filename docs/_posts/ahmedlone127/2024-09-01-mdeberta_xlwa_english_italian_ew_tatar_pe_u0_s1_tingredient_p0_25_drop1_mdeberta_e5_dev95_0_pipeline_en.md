---
layout: model
title: English mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline pipeline DeBertaForQuestionAnswering from pgajo
author: John Snow Labs
name: mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline` is a English model originally trained by pgajo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline_en_5.5.0_3.0_1725220949436.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline_en_5.5.0_3.0_1725220949436.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdeberta_xlwa_english_italian_ew_tatar_pe_u0_s1_tingredient_p0_25_drop1_mdeberta_e5_dev95_0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|817.8 MB|

## References

https://huggingface.co/pgajo/mdeberta-xlwa-en-it_EW-TT-PE_U0_S1_Tingredient_P0.25_DROP1_mdeberta_E5_DEV95.0

## Included Models

- MultiDocumentAssembler
- DeBertaForQuestionAnswering