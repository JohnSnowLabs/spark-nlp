---
layout: model
title: Arabic labour_law_sanskrit_saskta_qa_pipeline pipeline BertForQuestionAnswering from faisalaljahlan
author: John Snow Labs
name: labour_law_sanskrit_saskta_qa_pipeline
date: 2024-09-14
tags: [ar, open_source, pipeline, onnx]
task: Question Answering
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`labour_law_sanskrit_saskta_qa_pipeline` is a Arabic model originally trained by faisalaljahlan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/labour_law_sanskrit_saskta_qa_pipeline_ar_5.5.0_3.0_1726349646092.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/labour_law_sanskrit_saskta_qa_pipeline_ar_5.5.0_3.0_1726349646092.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("labour_law_sanskrit_saskta_qa_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("labour_law_sanskrit_saskta_qa_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|labour_law_sanskrit_saskta_qa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|504.6 MB|

## References

https://huggingface.co/faisalaljahlan/Labour-Law-SA-QA

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering