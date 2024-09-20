---
layout: model
title: Tagalog filipinolingo_translation_pipeline pipeline T5Transformer from youdiniplays
author: John Snow Labs
name: filipinolingo_translation_pipeline
date: 2024-08-24
tags: [tl, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: tl
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`filipinolingo_translation_pipeline` is a Tagalog model originally trained by youdiniplays.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/filipinolingo_translation_pipeline_tl_5.4.2_3.0_1724495665857.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/filipinolingo_translation_pipeline_tl_5.4.2_3.0_1724495665857.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("filipinolingo_translation_pipeline", lang = "tl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("filipinolingo_translation_pipeline", lang = "tl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|filipinolingo_translation_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|tl|
|Size:|316.5 MB|

## References

https://huggingface.co/youdiniplays/filipinolingo_translation

## Included Models

- DocumentAssembler
- T5Transformer