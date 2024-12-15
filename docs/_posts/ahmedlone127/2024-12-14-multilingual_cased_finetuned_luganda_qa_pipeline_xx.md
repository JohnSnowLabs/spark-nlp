---
layout: model
title: Multilingual multilingual_cased_finetuned_luganda_qa_pipeline pipeline BertForQuestionAnswering from badokorach
author: John Snow Labs
name: multilingual_cased_finetuned_luganda_qa_pipeline
date: 2024-12-14
tags: [xx, open_source, pipeline, onnx]
task: Question Answering
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilingual_cased_finetuned_luganda_qa_pipeline` is a Multilingual model originally trained by badokorach.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilingual_cased_finetuned_luganda_qa_pipeline_xx_5.5.1_3.0_1734215849499.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilingual_cased_finetuned_luganda_qa_pipeline_xx_5.5.1_3.0_1734215849499.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multilingual_cased_finetuned_luganda_qa_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multilingual_cased_finetuned_luganda_qa_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multilingual_cased_finetuned_luganda_qa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|665.1 MB|

## References

https://huggingface.co/badokorach/multilingual-cased-finetuned-luganda-qa

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering