---
layout: model
title: Finnish captaina_v0_pipeline pipeline Wav2Vec2ForCTC from Usin2705
author: John Snow Labs
name: captaina_v0_pipeline
date: 2025-04-07
tags: [fi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`captaina_v0_pipeline` is a Finnish model originally trained by Usin2705.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/captaina_v0_pipeline_fi_5.5.1_3.0_1744021735284.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/captaina_v0_pipeline_fi_5.5.1_3.0_1744021735284.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("captaina_v0_pipeline", lang = "fi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("captaina_v0_pipeline", lang = "fi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|captaina_v0_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|1.2 GB|

## References

https://huggingface.co/Usin2705/CaptainA_v0

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC