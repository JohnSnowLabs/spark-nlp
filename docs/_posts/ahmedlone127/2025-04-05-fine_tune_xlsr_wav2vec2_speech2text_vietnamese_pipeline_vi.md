---
layout: model
title: Vietnamese fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline pipeline Wav2Vec2ForCTC from leduytan93
author: John Snow Labs
name: fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline
date: 2025-04-05
tags: [vi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: vi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline` is a Vietnamese model originally trained by leduytan93.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline_vi_5.5.1_3.0_1743813747090.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline_vi_5.5.1_3.0_1743813747090.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|fine_tune_xlsr_wav2vec2_speech2text_vietnamese_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|1.2 GB|

## References

https://huggingface.co/leduytan93/Fine-Tune-XLSR-Wav2Vec2-Speech2Text-Vietnamese

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC