---
layout: model
title: English ote_dapt_camel_msa_hard_6580subsample_run1_pipeline pipeline BertForTokenClassification from salohnana2018
author: John Snow Labs
name: ote_dapt_camel_msa_hard_6580subsample_run1_pipeline
date: 2025-02-08
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ote_dapt_camel_msa_hard_6580subsample_run1_pipeline` is a English model originally trained by salohnana2018.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ote_dapt_camel_msa_hard_6580subsample_run1_pipeline_en_5.5.1_3.0_1739031509578.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ote_dapt_camel_msa_hard_6580subsample_run1_pipeline_en_5.5.1_3.0_1739031509578.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ote_dapt_camel_msa_hard_6580subsample_run1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ote_dapt_camel_msa_hard_6580subsample_run1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ote_dapt_camel_msa_hard_6580subsample_run1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.4 MB|

## References

https://huggingface.co/salohnana2018/OTE-DAPT-CAMEL-MSA-HARD-6580SUBSAMPLE-run1

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification