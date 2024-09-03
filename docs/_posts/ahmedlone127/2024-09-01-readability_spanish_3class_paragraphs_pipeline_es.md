---
layout: model
title: Castilian, Spanish readability_spanish_3class_paragraphs_pipeline pipeline RoBertaForSequenceClassification from somosnlp-hackathon-2022
author: John Snow Labs
name: readability_spanish_3class_paragraphs_pipeline
date: 2024-09-01
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`readability_spanish_3class_paragraphs_pipeline` is a Castilian, Spanish model originally trained by somosnlp-hackathon-2022.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/readability_spanish_3class_paragraphs_pipeline_es_5.4.2_3.0_1725195389419.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/readability_spanish_3class_paragraphs_pipeline_es_5.4.2_3.0_1725195389419.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("readability_spanish_3class_paragraphs_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("readability_spanish_3class_paragraphs_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|readability_spanish_3class_paragraphs_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|442.0 MB|

## References

https://huggingface.co/somosnlp-hackathon-2022/readability-es-3class-paragraphs

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification