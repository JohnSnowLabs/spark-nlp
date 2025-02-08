---
layout: model
title: Norwegian Nynorsk norwegian_bokml_wav2vec2_300m_nynorsk_pipeline pipeline Wav2Vec2ForCTC from NbAiLab
author: John Snow Labs
name: norwegian_bokml_wav2vec2_300m_nynorsk_pipeline
date: 2025-02-07
tags: [nn, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: nn
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`norwegian_bokml_wav2vec2_300m_nynorsk_pipeline` is a Norwegian Nynorsk model originally trained by NbAiLab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norwegian_bokml_wav2vec2_300m_nynorsk_pipeline_nn_5.5.1_3.0_1738908914779.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/norwegian_bokml_wav2vec2_300m_nynorsk_pipeline_nn_5.5.1_3.0_1738908914779.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("norwegian_bokml_wav2vec2_300m_nynorsk_pipeline", lang = "nn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("norwegian_bokml_wav2vec2_300m_nynorsk_pipeline", lang = "nn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|norwegian_bokml_wav2vec2_300m_nynorsk_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|nn|
|Size:|1.2 GB|

## References

https://huggingface.co/NbAiLab/nb-wav2vec2-300m-nynorsk

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC