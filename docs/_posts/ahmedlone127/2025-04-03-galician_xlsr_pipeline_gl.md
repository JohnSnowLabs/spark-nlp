---
layout: model
title: Galician galician_xlsr_pipeline pipeline Wav2Vec2ForCTC from Akashpb13
author: John Snow Labs
name: galician_xlsr_pipeline
date: 2025-04-03
tags: [gl, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: gl
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`galician_xlsr_pipeline` is a Galician model originally trained by Akashpb13.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/galician_xlsr_pipeline_gl_5.5.1_3.0_1743707248569.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/galician_xlsr_pipeline_gl_5.5.1_3.0_1743707248569.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("galician_xlsr_pipeline", lang = "gl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("galician_xlsr_pipeline", lang = "gl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|galician_xlsr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|gl|
|Size:|1.2 GB|

## References

https://huggingface.co/Akashpb13/Galician_xlsr

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC