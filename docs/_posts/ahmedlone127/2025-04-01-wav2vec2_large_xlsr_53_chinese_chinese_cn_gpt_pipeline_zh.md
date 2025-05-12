---
layout: model
title: Chinese wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline pipeline Wav2Vec2ForCTC from ydshieh
author: John Snow Labs
name: wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline
date: 2025-04-01
tags: [zh, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline` is a Chinese model originally trained by ydshieh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline_zh_5.5.1_3.0_1743543304217.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline_zh_5.5.1_3.0_1743543304217.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_large_xlsr_53_chinese_chinese_cn_gpt_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|1.3 GB|

## References

https://huggingface.co/ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC