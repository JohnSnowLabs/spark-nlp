---
layout: model
title: English afro_xlmr_base_hausa_seed_30_pipeline pipeline XlmRoBertaForTokenClassification from grace-pro
author: John Snow Labs
name: afro_xlmr_base_hausa_seed_30_pipeline
date: 2024-09-23
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`afro_xlmr_base_hausa_seed_30_pipeline` is a English model originally trained by grace-pro.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/afro_xlmr_base_hausa_seed_30_pipeline_en_5.5.0_3.0_1727061539034.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/afro_xlmr_base_hausa_seed_30_pipeline_en_5.5.0_3.0_1727061539034.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("afro_xlmr_base_hausa_seed_30_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("afro_xlmr_base_hausa_seed_30_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|afro_xlmr_base_hausa_seed_30_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/grace-pro/afro-xlmr-base-hausa-seed-30

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification