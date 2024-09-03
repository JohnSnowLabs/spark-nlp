---
layout: model
title: English sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline pipeline BertForQuestionAnswering from Kdogs
author: John Snow Labs
name: sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline
date: 2024-09-01
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline` is a English model originally trained by Kdogs.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline_en_5.5.0_3.0_1725215390546.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline_en_5.5.0_3.0_1725215390546.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sports_klue_finetuned_korquad_nosplit_nsp_batch32_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|412.4 MB|

## References

https://huggingface.co/Kdogs/sports_klue_finetuned_korquad_noSplit_NSP_batch32

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering