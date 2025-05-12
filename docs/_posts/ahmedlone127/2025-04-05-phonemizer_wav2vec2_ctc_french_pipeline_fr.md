---
layout: model
title: French phonemizer_wav2vec2_ctc_french_pipeline pipeline Wav2Vec2ForCTC from bofenghuang
author: John Snow Labs
name: phonemizer_wav2vec2_ctc_french_pipeline
date: 2025-04-05
tags: [fr, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`phonemizer_wav2vec2_ctc_french_pipeline` is a French model originally trained by bofenghuang.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/phonemizer_wav2vec2_ctc_french_pipeline_fr_5.5.1_3.0_1743840303151.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/phonemizer_wav2vec2_ctc_french_pipeline_fr_5.5.1_3.0_1743840303151.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("phonemizer_wav2vec2_ctc_french_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("phonemizer_wav2vec2_ctc_french_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|phonemizer_wav2vec2_ctc_french_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|1.2 GB|

## References

https://huggingface.co/bofenghuang/phonemizer-wav2vec2-ctc-french

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC