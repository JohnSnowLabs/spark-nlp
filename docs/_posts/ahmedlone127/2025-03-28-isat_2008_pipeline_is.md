---
layout: model
title: Icelandic isat_2008_pipeline pipeline XlmRoBertaForSequenceClassification from skatturinn
author: John Snow Labs
name: isat_2008_pipeline
date: 2025-03-28
tags: [is, open_source, pipeline, onnx]
task: Text Classification
language: is
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`isat_2008_pipeline` is a Icelandic model originally trained by skatturinn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/isat_2008_pipeline_is_5.5.1_3.0_1743153068412.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/isat_2008_pipeline_is_5.5.1_3.0_1743153068412.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("isat_2008_pipeline", lang = "is")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("isat_2008_pipeline", lang = "is")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|isat_2008_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|is|
|Size:|857.4 MB|

## References

https://huggingface.co/skatturinn/isat-2008

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification