---
layout: model
title: Castilian, Spanish location_sub_tagger_pipeline pipeline RoBertaForTokenClassification from BSC-NLP4BIA
author: John Snow Labs
name: location_sub_tagger_pipeline
date: 2025-03-27
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`location_sub_tagger_pipeline` is a Castilian, Spanish model originally trained by BSC-NLP4BIA.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/location_sub_tagger_pipeline_es_5.5.1_3.0_1743094041455.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/location_sub_tagger_pipeline_es_5.5.1_3.0_1743094041455.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("location_sub_tagger_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("location_sub_tagger_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|location_sub_tagger_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|449.1 MB|

## References

https://huggingface.co/BSC-NLP4BIA/location-sub-tagger

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification