---
layout: model
title: Gujarati gujarati_vdb_fine_tune_pipeline pipeline WhisperForCTC from iiBLACKii
author: John Snow Labs
name: gujarati_vdb_fine_tune_pipeline
date: 2024-12-17
tags: [gu, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: gu
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gujarati_vdb_fine_tune_pipeline` is a Gujarati model originally trained by iiBLACKii.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gujarati_vdb_fine_tune_pipeline_gu_5.5.1_3.0_1734403045883.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gujarati_vdb_fine_tune_pipeline_gu_5.5.1_3.0_1734403045883.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gujarati_vdb_fine_tune_pipeline", lang = "gu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gujarati_vdb_fine_tune_pipeline", lang = "gu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gujarati_vdb_fine_tune_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|gu|
|Size:|1.3 GB|

## References

https://huggingface.co/iiBLACKii/Gujarati_VDB_Fine_Tune

## Included Models

- AudioAssembler
- WhisperForCTC