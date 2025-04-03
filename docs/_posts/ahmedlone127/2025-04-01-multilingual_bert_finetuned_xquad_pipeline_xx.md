---
layout: model
title: Multilingual multilingual_bert_finetuned_xquad_pipeline pipeline BertForQuestionAnswering from ritwikm
author: John Snow Labs
name: multilingual_bert_finetuned_xquad_pipeline
date: 2025-04-01
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilingual_bert_finetuned_xquad_pipeline` is a Multilingual model originally trained by ritwikm.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilingual_bert_finetuned_xquad_pipeline_xx_5.5.1_3.0_1743516159633.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilingual_bert_finetuned_xquad_pipeline_xx_5.5.1_3.0_1743516159633.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multilingual_bert_finetuned_xquad_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multilingual_bert_finetuned_xquad_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multilingual_bert_finetuned_xquad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|665.1 MB|

## References

https://huggingface.co/ritwikm/multilingual-bert-finetuned-xquad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering