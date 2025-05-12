---
layout: model
title: Abkhazian hf_challenge_test_pipeline pipeline Wav2Vec2ForCTC from Iskaj
author: John Snow Labs
name: hf_challenge_test_pipeline
date: 2025-04-02
tags: [ab, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ab
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hf_challenge_test_pipeline` is a Abkhazian model originally trained by Iskaj.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hf_challenge_test_pipeline_ab_5.5.1_3.0_1743604303394.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hf_challenge_test_pipeline_ab_5.5.1_3.0_1743604303394.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hf_challenge_test_pipeline", lang = "ab")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hf_challenge_test_pipeline", lang = "ab")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hf_challenge_test_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ab|
|Size:|133.7 KB|

## References

https://huggingface.co/Iskaj/hf-challenge-test

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC