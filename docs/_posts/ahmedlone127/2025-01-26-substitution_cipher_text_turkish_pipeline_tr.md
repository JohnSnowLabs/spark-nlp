---
layout: model
title: Turkish substitution_cipher_text_turkish_pipeline pipeline T5Transformer from Cipher-AI
author: John Snow Labs
name: substitution_cipher_text_turkish_pipeline
date: 2025-01-26
tags: [tr, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: tr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`substitution_cipher_text_turkish_pipeline` is a Turkish model originally trained by Cipher-AI.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/substitution_cipher_text_turkish_pipeline_tr_5.5.1_3.0_1737850240910.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/substitution_cipher_text_turkish_pipeline_tr_5.5.1_3.0_1737850240910.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("substitution_cipher_text_turkish_pipeline", lang = "tr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("substitution_cipher_text_turkish_pipeline", lang = "tr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|substitution_cipher_text_turkish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|tr|
|Size:|927.2 MB|

## References

https://huggingface.co/Cipher-AI/Substitution-Cipher-Text-Turkish

## Included Models

- DocumentAssembler
- T5Transformer