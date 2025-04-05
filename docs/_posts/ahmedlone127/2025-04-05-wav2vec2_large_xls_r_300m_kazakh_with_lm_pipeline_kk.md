---
layout: model
title: Kazakh wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline pipeline Wav2Vec2ForCTC from DrishtiSharma
author: John Snow Labs
name: wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline
date: 2025-04-05
tags: [kk, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: kk
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline` is a Kazakh model originally trained by DrishtiSharma.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline_kk_5.5.1_3.0_1743815464150.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline_kk_5.5.1_3.0_1743815464150.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline", lang = "kk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline", lang = "kk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_large_xls_r_300m_kazakh_with_lm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|kk|
|Size:|1.2 GB|

## References

https://huggingface.co/DrishtiSharma/wav2vec2-large-xls-r-300m-kk-with-LM

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC