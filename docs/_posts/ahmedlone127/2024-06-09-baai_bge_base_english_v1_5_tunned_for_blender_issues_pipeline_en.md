---
layout: model
title: English baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline pipeline BGEEmbeddings from mano-wii
author: John Snow Labs
name: baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline
date: 2024-06-09
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline` is a English model originally trained by mano-wii.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline_en_5.4.0_3.0_1717956907363.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline_en_5.4.0_3.0_1717956907363.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|baai_bge_base_english_v1_5_tunned_for_blender_issues_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|391.8 MB|

## References

https://huggingface.co/mano-wii/BAAI_bge-base-en-v1.5-tunned-for-blender-issues

## Included Models

- DocumentAssembler
- BGEEmbeddings