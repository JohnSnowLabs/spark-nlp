---
layout: model
title: Castilian, Spanish roberta_qa_base_spanish_squades_becasv3_pipeline pipeline RoBertaForQuestionAnswering from Evelyn18
author: John Snow Labs
name: roberta_qa_base_spanish_squades_becasv3_pipeline
date: 2024-09-01
tags: [es, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_qa_base_spanish_squades_becasv3_pipeline` is a Castilian, Spanish model originally trained by Evelyn18.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_base_spanish_squades_becasv3_pipeline_es_5.4.2_3.0_1725200446177.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_qa_base_spanish_squades_becasv3_pipeline_es_5.4.2_3.0_1725200446177.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_qa_base_spanish_squades_becasv3_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_qa_base_spanish_squades_becasv3_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_qa_base_spanish_squades_becasv3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|459.1 MB|

## References

https://huggingface.co/Evelyn18/roberta-base-spanish-squades-becasv3

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering