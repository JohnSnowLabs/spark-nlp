---
layout: model
title: Tamil tamil_wav2vec_xls_r_300m_tamil_colab_pipeline pipeline Wav2Vec2ForCTC from bharat-raghunathan
author: John Snow Labs
name: tamil_wav2vec_xls_r_300m_tamil_colab_pipeline
date: 2025-04-07
tags: [ta, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ta
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tamil_wav2vec_xls_r_300m_tamil_colab_pipeline` is a Tamil model originally trained by bharat-raghunathan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tamil_wav2vec_xls_r_300m_tamil_colab_pipeline_ta_5.5.1_3.0_1744021741937.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tamil_wav2vec_xls_r_300m_tamil_colab_pipeline_ta_5.5.1_3.0_1744021741937.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tamil_wav2vec_xls_r_300m_tamil_colab_pipeline", lang = "ta")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tamil_wav2vec_xls_r_300m_tamil_colab_pipeline", lang = "ta")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tamil_wav2vec_xls_r_300m_tamil_colab_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ta|
|Size:|1.2 GB|

## References

https://huggingface.co/bharat-raghunathan/Tamil-Wav2Vec-xls-r-300m-Tamil-colab

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC