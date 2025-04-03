---
layout: model
title: Estonian xls_r_300m_estonian_pipeline pipeline Wav2Vec2ForCTC from TalTechNLP
author: John Snow Labs
name: xls_r_300m_estonian_pipeline
date: 2025-04-03
tags: [et, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: et
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xls_r_300m_estonian_pipeline` is a Estonian model originally trained by TalTechNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xls_r_300m_estonian_pipeline_et_5.5.1_3.0_1743706940269.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xls_r_300m_estonian_pipeline_et_5.5.1_3.0_1743706940269.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xls_r_300m_estonian_pipeline", lang = "et")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xls_r_300m_estonian_pipeline", lang = "et")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xls_r_300m_estonian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|et|
|Size:|764.3 MB|

## References

https://huggingface.co/TalTechNLP/xls-r-300m-et

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC