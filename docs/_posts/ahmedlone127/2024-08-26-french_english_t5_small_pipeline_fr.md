---
layout: model
title: French french_english_t5_small_pipeline pipeline T5Transformer from Korventenn
author: John Snow Labs
name: french_english_t5_small_pipeline
date: 2024-08-26
tags: [fr, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: fr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`french_english_t5_small_pipeline` is a French model originally trained by Korventenn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/french_english_t5_small_pipeline_fr_5.4.2_3.0_1724641128710.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/french_english_t5_small_pipeline_fr_5.4.2_3.0_1724641128710.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("french_english_t5_small_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("french_english_t5_small_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|french_english_t5_small_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|247.2 MB|

## References

https://huggingface.co/Korventenn/fr_en-t5-small

## Included Models

- DocumentAssembler
- T5Transformer