---
layout: model
title: English minilmv2_l6_h384_from_bert_large_mrqa_pipeline pipeline BertForQuestionAnswering from VMware
author: John Snow Labs
name: minilmv2_l6_h384_from_bert_large_mrqa_pipeline
date: 2024-09-22
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`minilmv2_l6_h384_from_bert_large_mrqa_pipeline` is a English model originally trained by VMware.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/minilmv2_l6_h384_from_bert_large_mrqa_pipeline_en_5.5.0_3.0_1726991824080.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/minilmv2_l6_h384_from_bert_large_mrqa_pipeline_en_5.5.0_3.0_1726991824080.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("minilmv2_l6_h384_from_bert_large_mrqa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("minilmv2_l6_h384_from_bert_large_mrqa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|minilmv2_l6_h384_from_bert_large_mrqa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|84.3 MB|

## References

https://huggingface.co/VMware/minilmv2-l6-h384-from-bert-large-mrqa

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering