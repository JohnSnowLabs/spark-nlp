---
layout: model
title: Indonesian wav2vec2_large_xlsr_53_indonesian_pipeline pipeline Wav2Vec2ForCTC from arifagustyawan
author: John Snow Labs
name: wav2vec2_large_xlsr_53_indonesian_pipeline
date: 2025-03-31
tags: [id, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xlsr_53_indonesian_pipeline` is a Indonesian model originally trained by arifagustyawan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_indonesian_pipeline_id_5.5.1_3.0_1743450378201.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_indonesian_pipeline_id_5.5.1_3.0_1743450378201.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_large_xlsr_53_indonesian_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_large_xlsr_53_indonesian_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_large_xlsr_53_indonesian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|1.2 GB|

## References

https://huggingface.co/arifagustyawan/wav2vec2-large-xlsr-53-id

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC