---
layout: model
title: Korean bert_base_klue_mrc_finetuned_pipeline pipeline BertForQuestionAnswering from JiHoon-kim
author: John Snow Labs
name: bert_base_klue_mrc_finetuned_pipeline
date: 2025-04-04
tags: [ko, open_source, pipeline, onnx]
task: Question Answering
language: ko
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_klue_mrc_finetuned_pipeline` is a Korean model originally trained by JiHoon-kim.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_klue_mrc_finetuned_pipeline_ko_5.5.1_3.0_1743759694060.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_klue_mrc_finetuned_pipeline_ko_5.5.1_3.0_1743759694060.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_klue_mrc_finetuned_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_klue_mrc_finetuned_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_klue_mrc_finetuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|412.4 MB|

## References

https://huggingface.co/JiHoon-kim/bert-base-klue-mrc-finetuned

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering