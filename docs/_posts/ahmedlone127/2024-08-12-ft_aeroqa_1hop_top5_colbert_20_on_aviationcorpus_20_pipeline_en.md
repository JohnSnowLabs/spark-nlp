---
layout: model
title: English ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline pipeline T5Transformer from sakharamg
author: John Snow Labs
name: ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline
date: 2024-08-12
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline` is a English model originally trained by sakharamg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline_en_5.4.2_3.0_1723489004908.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline_en_5.4.2_3.0_1723489004908.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ft_aeroqa_1hop_top5_colbert_20_on_aviationcorpus_20_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|3.0 GB|

## References

https://huggingface.co/sakharamg/FT_aeroqa_1hop_top5_COLBERT_20_on_aviationcorpus_20

## Included Models

- DocumentAssembler
- T5Transformer