---
layout: model
title: Galician nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline pipeline Wav2Vec2ForCTC from proxectonos
author: John Snow Labs
name: nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline
date: 2025-03-31
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline` is a Galician model originally trained by proxectonos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline_gl_5.5.1_3.0_1743425500246.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline_gl_5.5.1_3.0_1743425500246.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline", lang = "gl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline", lang = "gl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nos_asr_wav2vec2_large_xlsr_53_galician_with_lm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|gl|
|Size:|1.2 GB|

## References

https://huggingface.co/proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC