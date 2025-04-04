---
layout: model
title: German wav2vec2_base_10k_voxpopuli_ft_german_pipeline pipeline Wav2Vec2ForCTC from facebook
author: John Snow Labs
name: wav2vec2_base_10k_voxpopuli_ft_german_pipeline
date: 2025-04-04
tags: [de, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: de
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_base_10k_voxpopuli_ft_german_pipeline` is a German model originally trained by facebook.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_base_10k_voxpopuli_ft_german_pipeline_de_5.5.1_3.0_1743763929966.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_base_10k_voxpopuli_ft_german_pipeline_de_5.5.1_3.0_1743763929966.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_base_10k_voxpopuli_ft_german_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_base_10k_voxpopuli_ft_german_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_base_10k_voxpopuli_ft_german_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|233.7 MB|

## References

https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli-ft-de

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC