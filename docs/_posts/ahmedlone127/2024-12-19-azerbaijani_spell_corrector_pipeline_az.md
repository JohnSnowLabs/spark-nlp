---
layout: model
title: Azerbaijani azerbaijani_spell_corrector_pipeline pipeline T5Transformer from LocalDoc
author: John Snow Labs
name: azerbaijani_spell_corrector_pipeline
date: 2024-12-19
tags: [az, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: az
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`azerbaijani_spell_corrector_pipeline` is a Azerbaijani model originally trained by LocalDoc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/azerbaijani_spell_corrector_pipeline_az_5.5.1_3.0_1734568700753.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/azerbaijani_spell_corrector_pipeline_az_5.5.1_3.0_1734568700753.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("azerbaijani_spell_corrector_pipeline", lang = "az")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("azerbaijani_spell_corrector_pipeline", lang = "az")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|azerbaijani_spell_corrector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|az|
|Size:|1.3 GB|

## References

https://huggingface.co/LocalDoc/azerbaijani_spell_corrector

## Included Models

- DocumentAssembler
- T5Transformer