---
layout: model
title: English xls_1b_decoding_french_decoding_pipeline pipeline Wav2Vec2ForCTC from Dandan0K
author: John Snow Labs
name: xls_1b_decoding_french_decoding_pipeline
date: 2025-03-29
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xls_1b_decoding_french_decoding_pipeline` is a English model originally trained by Dandan0K.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xls_1b_decoding_french_decoding_pipeline_en_5.5.1_3.0_1743287121452.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xls_1b_decoding_french_decoding_pipeline_en_5.5.1_3.0_1743287121452.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xls_1b_decoding_french_decoding_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xls_1b_decoding_french_decoding_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xls_1b_decoding_french_decoding_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/Dandan0K/xls_1b_decoding_fr_decoding

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC