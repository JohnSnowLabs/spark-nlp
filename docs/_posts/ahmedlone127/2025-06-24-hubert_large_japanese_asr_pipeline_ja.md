---
layout: model
title: Japanese hubert_large_japanese_asr_pipeline pipeline HubertForCTC from TKU410410103
author: John Snow Labs
name: hubert_large_japanese_asr_pipeline
date: 2025-06-24
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

Pretrained HubertForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hubert_large_japanese_asr_pipeline` is a Japanese model originally trained by TKU410410103.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hubert_large_japanese_asr_pipeline_ja_5.5.1_3.0_1750783786488.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hubert_large_japanese_asr_pipeline_ja_5.5.1_3.0_1750783786488.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("hubert_large_japanese_asr_pipeline", lang = "ja")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("hubert_large_japanese_asr_pipeline", lang = "ja")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hubert_large_japanese_asr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ja|
|Size:|2.4 GB|

## References

References

https://huggingface.co/TKU410410103/hubert-large-japanese-asr

## Included Models

- AudioAssembler
- HubertForCTC