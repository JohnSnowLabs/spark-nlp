---
layout: model
title: Nepali (macrolanguage) wav2vec2_large_xlsr_300m_nepali_pipeline pipeline Wav2Vec2ForCTC from prajin
author: John Snow Labs
name: wav2vec2_large_xlsr_300m_nepali_pipeline
date: 2025-03-31
tags: [ne, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ne
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xlsr_300m_nepali_pipeline` is a Nepali (macrolanguage) model originally trained by prajin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_300m_nepali_pipeline_ne_5.5.1_3.0_1743424709820.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_300m_nepali_pipeline_ne_5.5.1_3.0_1743424709820.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_large_xlsr_300m_nepali_pipeline", lang = "ne")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_large_xlsr_300m_nepali_pipeline", lang = "ne")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_large_xlsr_300m_nepali_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ne|
|Size:|1.2 GB|

## References

https://huggingface.co/prajin/wav2vec2-large-xlsr-300m-nepali

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC