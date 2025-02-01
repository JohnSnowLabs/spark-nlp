---
layout: model
title: French lemone_router_m_pipeline pipeline XlmRoBertaForSequenceClassification from louisbrulenaudet
author: John Snow Labs
name: lemone_router_m_pipeline
date: 2025-01-29
tags: [fr, open_source, pipeline, onnx]
task: Text Classification
language: fr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`lemone_router_m_pipeline` is a French model originally trained by louisbrulenaudet.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/lemone_router_m_pipeline_fr_5.5.1_3.0_1738178664393.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/lemone_router_m_pipeline_fr_5.5.1_3.0_1738178664393.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("lemone_router_m_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("lemone_router_m_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|lemone_router_m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|851.6 MB|

## References

https://huggingface.co/louisbrulenaudet/lemone-router-m

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification