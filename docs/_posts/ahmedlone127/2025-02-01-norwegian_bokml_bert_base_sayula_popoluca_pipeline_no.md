---
layout: model
title: Norwegian norwegian_bokml_bert_base_sayula_popoluca_pipeline pipeline BertForTokenClassification from NbAiLab
author: John Snow Labs
name: norwegian_bokml_bert_base_sayula_popoluca_pipeline
date: 2025-02-01
tags: ["no", open_source, pipeline, onnx]
task: Named Entity Recognition
language: "no"
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`norwegian_bokml_bert_base_sayula_popoluca_pipeline` is a Norwegian model originally trained by NbAiLab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/norwegian_bokml_bert_base_sayula_popoluca_pipeline_no_5.5.1_3.0_1738381700434.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/norwegian_bokml_bert_base_sayula_popoluca_pipeline_no_5.5.1_3.0_1738381700434.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("norwegian_bokml_bert_base_sayula_popoluca_pipeline", lang = "no")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("norwegian_bokml_bert_base_sayula_popoluca_pipeline", lang = "no")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|norwegian_bokml_bert_base_sayula_popoluca_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|no|
|Size:|666.3 MB|

## References

https://huggingface.co/NbAiLab/nb-bert-base-pos

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification