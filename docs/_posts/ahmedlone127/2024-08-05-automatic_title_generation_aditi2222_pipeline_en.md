---
layout: model
title: English automatic_title_generation_aditi2222_pipeline pipeline T5Transformer from aditi2222
author: John Snow Labs
name: automatic_title_generation_aditi2222_pipeline
date: 2024-08-05
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`automatic_title_generation_aditi2222_pipeline` is a English model originally trained by aditi2222.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/automatic_title_generation_aditi2222_pipeline_en_5.4.2_3.0_1722820601813.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/automatic_title_generation_aditi2222_pipeline_en_5.4.2_3.0_1722820601813.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("automatic_title_generation_aditi2222_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("automatic_title_generation_aditi2222_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|automatic_title_generation_aditi2222_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|993.0 MB|

## References

https://huggingface.co/aditi2222/automatic_title_generation

## Included Models

- DocumentAssembler
- T5Transformer