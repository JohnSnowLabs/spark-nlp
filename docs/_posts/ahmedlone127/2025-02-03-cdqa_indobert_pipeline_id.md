---
layout: model
title: Indonesian cdqa_indobert_pipeline pipeline BertForQuestionAnswering from emny
author: John Snow Labs
name: cdqa_indobert_pipeline
date: 2025-02-03
tags: [id, open_source, pipeline, onnx]
task: Question Answering
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cdqa_indobert_pipeline` is a Indonesian model originally trained by emny.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cdqa_indobert_pipeline_id_5.5.1_3.0_1738558607463.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cdqa_indobert_pipeline_id_5.5.1_3.0_1738558607463.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cdqa_indobert_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cdqa_indobert_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cdqa_indobert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|411.7 MB|

## References

https://huggingface.co/emny/cdqa-indobert

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering