---
layout: model
title: Norwegian norwegian_bokml_bert_base_qa_norquad_pipeline pipeline BertForQuestionAnswering from eanderson
author: John Snow Labs
name: norwegian_bokml_bert_base_qa_norquad_pipeline
date: 2024-09-01
tags: ["no", open_source, pipeline, onnx]
task: Question Answering
language: "no"
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`norwegian_bokml_bert_base_qa_norquad_pipeline` is a Norwegian model originally trained by eanderson.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norwegian_bokml_bert_base_qa_norquad_pipeline_no_5.5.0_3.0_1725215063931.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/norwegian_bokml_bert_base_qa_norquad_pipeline_no_5.5.0_3.0_1725215063931.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("norwegian_bokml_bert_base_qa_norquad_pipeline", lang = "no")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("norwegian_bokml_bert_base_qa_norquad_pipeline", lang = "no")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|norwegian_bokml_bert_base_qa_norquad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|no|
|Size:|666.2 MB|

## References

https://huggingface.co/eanderson/nb-bert-base-qa-norquad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering