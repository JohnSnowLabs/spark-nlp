---
layout: model
title: Japanese hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline pipeline HubertForCTC from utakumi
author: John Snow Labs
name: hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline
date: 2025-01-23
tags: [ja, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ja
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained HubertForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline` is a Japanese model originally trained by utakumi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline_ja_5.5.1_3.0_1737625877844.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline_ja_5.5.1_3.0_1737625877844.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline", lang = "ja")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline", lang = "ja")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hubert_common_voice_japanese_demo_phonemes_cosine_3e_5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|698.0 MB|

## References

https://huggingface.co/utakumi/Hubert-common_voice-ja-demo-phonemes-cosine-3e-5

## Included Models

- AudioAssembler
- HubertForCTC