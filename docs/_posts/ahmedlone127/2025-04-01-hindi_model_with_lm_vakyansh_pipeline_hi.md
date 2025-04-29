---
layout: model
title: Hindi hindi_model_with_lm_vakyansh_pipeline pipeline Wav2Vec2ForCTC from Harveenchadha
author: John Snow Labs
name: hindi_model_with_lm_vakyansh_pipeline
date: 2025-04-01
tags: [hi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: hi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hindi_model_with_lm_vakyansh_pipeline` is a Hindi model originally trained by Harveenchadha.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hindi_model_with_lm_vakyansh_pipeline_hi_5.5.1_3.0_1743542514853.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hindi_model_with_lm_vakyansh_pipeline_hi_5.5.1_3.0_1743542514853.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hindi_model_with_lm_vakyansh_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hindi_model_with_lm_vakyansh_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hindi_model_with_lm_vakyansh_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|233.4 MB|

## References

https://huggingface.co/Harveenchadha/hindi_model_with_lm_vakyansh

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC