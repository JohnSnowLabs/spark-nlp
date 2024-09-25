---
layout: model
title: Korean bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline pipeline BertForQuestionAnswering from jihoonkimharu
author: John Snow Labs
name: bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline
date: 2024-09-20
tags: [ko, open_source, pipeline, onnx]
task: Question Answering
language: ko
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline` is a Korean model originally trained by jihoonkimharu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline_ko_5.5.0_3.0_1726820662656.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline_ko_5.5.0_3.0_1726820662656.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_klue_mrc_finetuned_jihoonkimharu_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|412.4 MB|

## References

https://huggingface.co/jihoonkimharu/bert-base-klue-mrc-finetuned

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering