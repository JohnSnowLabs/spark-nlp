---
layout: model
title: Hungarian whisper_tiny_cv16_hungarian_final_pipeline pipeline WhisperForCTC from Hungarians
author: John Snow Labs
name: whisper_tiny_cv16_hungarian_final_pipeline
date: 2024-09-21
tags: [hu, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: hu
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_cv16_hungarian_final_pipeline` is a Hungarian model originally trained by Hungarians.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_cv16_hungarian_final_pipeline_hu_5.5.0_3.0_1726892615254.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_cv16_hungarian_final_pipeline_hu_5.5.0_3.0_1726892615254.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_cv16_hungarian_final_pipeline", lang = "hu")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_cv16_hungarian_final_pipeline", lang = "hu")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_cv16_hungarian_final_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|hu|
|Size:|385.4 MB|

## References

https://huggingface.co/Hungarians/whisper-tiny-cv16-hu-final

## Included Models

- AudioAssembler
- WhisperForCTC