---
layout: model
title: Finnish ul2_small_nl16_finnish_pipeline pipeline T5Transformer from Finnish-NLP
author: John Snow Labs
name: ul2_small_nl16_finnish_pipeline
date: 2024-08-09
tags: [fi, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: fi
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ul2_small_nl16_finnish_pipeline` is a Finnish model originally trained by Finnish-NLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ul2_small_nl16_finnish_pipeline_fi_5.4.2_3.0_1723198384605.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ul2_small_nl16_finnish_pipeline_fi_5.4.2_3.0_1723198384605.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ul2_small_nl16_finnish_pipeline", lang = "fi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ul2_small_nl16_finnish_pipeline", lang = "fi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ul2_small_nl16_finnish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|750.9 MB|

## References

https://huggingface.co/Finnish-NLP/ul2-small-nl16-finnish

## Included Models

- DocumentAssembler
- T5Transformer