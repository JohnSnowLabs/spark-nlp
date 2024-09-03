---
layout: model
title: English xdoc_base_funsd_pipeline pipeline RoBertaForTokenClassification from microsoft
author: John Snow Labs
name: xdoc_base_funsd_pipeline
date: 2024-09-03
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xdoc_base_funsd_pipeline` is a English model originally trained by microsoft.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xdoc_base_funsd_pipeline_en_5.5.0_3.0_1725383691242.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xdoc_base_funsd_pipeline_en_5.5.0_3.0_1725383691242.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xdoc_base_funsd_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xdoc_base_funsd_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xdoc_base_funsd_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.2 MB|

## References

https://huggingface.co/microsoft/xdoc-base-funsd

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification