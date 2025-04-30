---
layout: model
title: Oriya (macrolanguage) indicwav2vec_odia_pipeline pipeline Wav2Vec2ForCTC from ai4bharat
author: John Snow Labs
name: indicwav2vec_odia_pipeline
date: 2025-04-02
tags: [or, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: or
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`indicwav2vec_odia_pipeline` is a Oriya (macrolanguage) model originally trained by ai4bharat.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/indicwav2vec_odia_pipeline_or_5.5.1_3.0_1743591484391.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/indicwav2vec_odia_pipeline_or_5.5.1_3.0_1743591484391.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("indicwav2vec_odia_pipeline", lang = "or")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("indicwav2vec_odia_pipeline", lang = "or")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|indicwav2vec_odia_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|or|
|Size:|771.2 MB|

## References

https://huggingface.co/ai4bharat/indicwav2vec-odia

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC