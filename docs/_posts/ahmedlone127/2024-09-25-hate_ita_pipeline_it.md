---
layout: model
title: Italian hate_ita_pipeline pipeline XlmRoBertaForSequenceClassification from MilaNLProc
author: John Snow Labs
name: hate_ita_pipeline
date: 2024-09-25
tags: [it, open_source, pipeline, onnx]
task: Text Classification
language: it
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hate_ita_pipeline` is a Italian model originally trained by MilaNLProc.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hate_ita_pipeline_it_5.5.0_3.0_1727229711840.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hate_ita_pipeline_it_5.5.0_3.0_1727229711840.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hate_ita_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hate_ita_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hate_ita_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|1.0 GB|

## References

https://huggingface.co/MilaNLProc/hate-ita

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification