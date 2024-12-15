---
layout: model
title: English distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline pipeline DistilBertForQuestionAnswering from Z-Jafari
author: John Snow Labs
name: distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline
date: 2024-12-14
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline` is a English model originally trained by Z-Jafari.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline_en_5.5.1_3.0_1734219415091.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline_en_5.5.1_3.0_1734219415091.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_persian_farsi_zwnj_base_finetuned_pquad_persianquad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|282.3 MB|

## References

https://huggingface.co/Z-Jafari/distilbert-fa-zwnj-base-finetuned-pquad-PersianQuAD

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering