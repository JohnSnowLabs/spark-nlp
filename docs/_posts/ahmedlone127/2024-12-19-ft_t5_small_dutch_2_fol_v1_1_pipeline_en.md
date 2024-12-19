---
layout: model
title: English ft_t5_small_dutch_2_fol_v1_1_pipeline pipeline T5Transformer from tuanbc88
author: John Snow Labs
name: ft_t5_small_dutch_2_fol_v1_1_pipeline
date: 2024-12-19
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ft_t5_small_dutch_2_fol_v1_1_pipeline` is a English model originally trained by tuanbc88.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ft_t5_small_dutch_2_fol_v1_1_pipeline_en_5.5.1_3.0_1734567788792.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ft_t5_small_dutch_2_fol_v1_1_pipeline_en_5.5.1_3.0_1734567788792.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ft_t5_small_dutch_2_fol_v1_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ft_t5_small_dutch_2_fol_v1_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ft_t5_small_dutch_2_fol_v1_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|147.3 MB|

## References

https://huggingface.co/tuanbc88/ft-t5-small-nl-2-fol-v1.1

## Included Models

- DocumentAssembler
- T5Transformer