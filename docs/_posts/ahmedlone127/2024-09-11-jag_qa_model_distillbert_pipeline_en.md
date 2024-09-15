---
layout: model
title: English jag_qa_model_distillbert_pipeline pipeline DistilBertForQuestionAnswering from jagmohanntnuiiit
author: John Snow Labs
name: jag_qa_model_distillbert_pipeline
date: 2024-09-11
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`jag_qa_model_distillbert_pipeline` is a English model originally trained by jagmohanntnuiiit.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/jag_qa_model_distillbert_pipeline_en_5.5.0_3.0_1726087974385.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/jag_qa_model_distillbert_pipeline_en_5.5.0_3.0_1726087974385.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("jag_qa_model_distillbert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("jag_qa_model_distillbert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jag_qa_model_distillbert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/jagmohanntnuiiit/jag_qa_model_distillbert

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering