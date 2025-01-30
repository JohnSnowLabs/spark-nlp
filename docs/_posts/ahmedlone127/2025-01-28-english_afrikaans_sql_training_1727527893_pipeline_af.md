---
layout: model
title: Afrikaans english_afrikaans_sql_training_1727527893_pipeline pipeline T5Transformer from JsteReubsSoftware
author: John Snow Labs
name: english_afrikaans_sql_training_1727527893_pipeline
date: 2025-01-28
tags: [af, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: af
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`english_afrikaans_sql_training_1727527893_pipeline` is a Afrikaans model originally trained by JsteReubsSoftware.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/english_afrikaans_sql_training_1727527893_pipeline_af_5.5.1_3.0_1738032626187.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/english_afrikaans_sql_training_1727527893_pipeline_af_5.5.1_3.0_1738032626187.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("english_afrikaans_sql_training_1727527893_pipeline", lang = "af")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("english_afrikaans_sql_training_1727527893_pipeline", lang = "af")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|english_afrikaans_sql_training_1727527893_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|af|
|Size:|147.7 MB|

## References

https://huggingface.co/JsteReubsSoftware/en-af-sql-training-1727527893

## Included Models

- DocumentAssembler
- T5Transformer