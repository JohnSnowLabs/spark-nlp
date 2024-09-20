---
layout: model
title: Castilian, Spanish roberta_base_biomedical_spanish_squad2_spanish_pipeline pipeline RoBertaForQuestionAnswering from somosnlp-hackathon-2022
author: John Snow Labs
name: roberta_base_biomedical_spanish_squad2_spanish_pipeline
date: 2024-09-03
tags: [es, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_biomedical_spanish_squad2_spanish_pipeline` is a Castilian, Spanish model originally trained by somosnlp-hackathon-2022.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_biomedical_spanish_squad2_spanish_pipeline_es_5.5.0_3.0_1725371171564.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_biomedical_spanish_squad2_spanish_pipeline_es_5.5.0_3.0_1725371171564.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_biomedical_spanish_squad2_spanish_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_biomedical_spanish_squad2_spanish_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_biomedical_spanish_squad2_spanish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|464.7 MB|

## References

https://huggingface.co/somosnlp-hackathon-2022/roberta-base-biomedical-es-squad2-es

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering