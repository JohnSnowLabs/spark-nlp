---
layout: model
title: Wolof marone_wolof_wav2vec2_xls_r_300m_pipeline pipeline Wav2Vec2ForCTC from M9and2M
author: John Snow Labs
name: marone_wolof_wav2vec2_xls_r_300m_pipeline
date: 2025-04-09
tags: [wo, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: wo
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marone_wolof_wav2vec2_xls_r_300m_pipeline` is a Wolof model originally trained by M9and2M.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marone_wolof_wav2vec2_xls_r_300m_pipeline_wo_5.5.1_3.0_1744193125589.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marone_wolof_wav2vec2_xls_r_300m_pipeline_wo_5.5.1_3.0_1744193125589.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marone_wolof_wav2vec2_xls_r_300m_pipeline", lang = "wo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marone_wolof_wav2vec2_xls_r_300m_pipeline", lang = "wo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marone_wolof_wav2vec2_xls_r_300m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|wo|
|Size:|1.2 GB|

## References

https://huggingface.co/M9and2M/marone_wolof_wav2vec2-xls-r-300m

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC