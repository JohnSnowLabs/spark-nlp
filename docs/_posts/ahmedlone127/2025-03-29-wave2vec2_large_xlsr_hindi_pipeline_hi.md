---
layout: model
title: Hindi wave2vec2_large_xlsr_hindi_pipeline pipeline Wav2Vec2ForCTC from shiwangi27
author: John Snow Labs
name: wave2vec2_large_xlsr_hindi_pipeline
date: 2025-03-29
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

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wave2vec2_large_xlsr_hindi_pipeline` is a Hindi model originally trained by shiwangi27.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wave2vec2_large_xlsr_hindi_pipeline_hi_5.5.1_3.0_1743216580348.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wave2vec2_large_xlsr_hindi_pipeline_hi_5.5.1_3.0_1743216580348.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wave2vec2_large_xlsr_hindi_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wave2vec2_large_xlsr_hindi_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wave2vec2_large_xlsr_hindi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|1.2 GB|

## References

https://huggingface.co/shiwangi27/wave2vec2-large-xlsr-hindi

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC