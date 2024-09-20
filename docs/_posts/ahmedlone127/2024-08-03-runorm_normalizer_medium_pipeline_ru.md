---
layout: model
title: Russian runorm_normalizer_medium_pipeline pipeline T5Transformer from RUNorm
author: John Snow Labs
name: runorm_normalizer_medium_pipeline
date: 2024-08-03
tags: [ru, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: ru
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`runorm_normalizer_medium_pipeline` is a Russian model originally trained by RUNorm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/runorm_normalizer_medium_pipeline_ru_5.4.2_3.0_1722659884821.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/runorm_normalizer_medium_pipeline_ru_5.4.2_3.0_1722659884821.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("runorm_normalizer_medium_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("runorm_normalizer_medium_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|runorm_normalizer_medium_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|520.8 MB|

## References

https://huggingface.co/RUNorm/RUNorm-normalizer-medium

## Included Models

- DocumentAssembler
- T5Transformer