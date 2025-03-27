---
layout: model
title: Bengali finetune_wav2vec2_large_xlsr_bengali_pipeline pipeline Wav2Vec2ForCTC from sshasnain
author: John Snow Labs
name: finetune_wav2vec2_large_xlsr_bengali_pipeline
date: 2025-03-27
tags: [bn, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: bn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetune_wav2vec2_large_xlsr_bengali_pipeline` is a Bengali model originally trained by sshasnain.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetune_wav2vec2_large_xlsr_bengali_pipeline_bn_5.5.1_3.0_1743078643161.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetune_wav2vec2_large_xlsr_bengali_pipeline_bn_5.5.1_3.0_1743078643161.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetune_wav2vec2_large_xlsr_bengali_pipeline", lang = "bn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetune_wav2vec2_large_xlsr_bengali_pipeline", lang = "bn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetune_wav2vec2_large_xlsr_bengali_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|bn|
|Size:|1.2 GB|

## References

https://huggingface.co/sshasnain/finetune-wav2vec2-large-xlsr-bengali

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC