---
layout: model
title: Mongolian wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline pipeline Wav2Vec2ForCTC from tugstugi
author: John Snow Labs
name: wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline
date: 2025-03-27
tags: [mn, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: mn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline` is a Mongolian model originally trained by tugstugi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline_mn_5.5.1_3.0_1743078538231.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline_mn_5.5.1_3.0_1743078538231.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline", lang = "mn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline", lang = "mn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_large_xlsr_53_mongolian_tugstugi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|mn|
|Size:|1.2 GB|

## References

https://huggingface.co/tugstugi/wav2vec2-large-xlsr-53-mongolian

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC