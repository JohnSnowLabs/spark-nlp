---
layout: model
title: Castilian, Spanish bsc_bio_ehr_spanish_carmen_procedimiento_pipeline pipeline RoBertaForTokenClassification from BSC-NLP4BIA
author: John Snow Labs
name: bsc_bio_ehr_spanish_carmen_procedimiento_pipeline
date: 2024-09-17
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bsc_bio_ehr_spanish_carmen_procedimiento_pipeline` is a Castilian, Spanish model originally trained by BSC-NLP4BIA.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bsc_bio_ehr_spanish_carmen_procedimiento_pipeline_es_5.5.0_3.0_1726538162055.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bsc_bio_ehr_spanish_carmen_procedimiento_pipeline_es_5.5.0_3.0_1726538162055.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bsc_bio_ehr_spanish_carmen_procedimiento_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bsc_bio_ehr_spanish_carmen_procedimiento_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bsc_bio_ehr_spanish_carmen_procedimiento_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|437.7 MB|

## References

https://huggingface.co/BSC-NLP4BIA/bsc-bio-ehr-es-carmen-procedimiento

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification