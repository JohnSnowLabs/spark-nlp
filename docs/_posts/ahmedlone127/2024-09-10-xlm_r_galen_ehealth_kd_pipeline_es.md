---
layout: model
title: Castilian, Spanish xlm_r_galen_ehealth_kd_pipeline pipeline XlmRoBertaForTokenClassification from IIC
author: John Snow Labs
name: xlm_r_galen_ehealth_kd_pipeline
date: 2024-09-10
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_r_galen_ehealth_kd_pipeline` is a Castilian, Spanish model originally trained by IIC.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_r_galen_ehealth_kd_pipeline_es_5.5.0_3.0_1725972881446.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_r_galen_ehealth_kd_pipeline_es_5.5.0_3.0_1725972881446.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_r_galen_ehealth_kd_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_r_galen_ehealth_kd_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_r_galen_ehealth_kd_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|1.0 GB|

## References

https://huggingface.co/IIC/XLM_R_Galen-ehealth_kd

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification