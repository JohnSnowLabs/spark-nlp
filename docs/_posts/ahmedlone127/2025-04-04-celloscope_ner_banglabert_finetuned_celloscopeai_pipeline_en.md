---
layout: model
title: English celloscope_ner_banglabert_finetuned_celloscopeai_pipeline pipeline BertForTokenClassification from celloscopeai
author: John Snow Labs
name: celloscope_ner_banglabert_finetuned_celloscopeai_pipeline
date: 2025-04-04
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`celloscope_ner_banglabert_finetuned_celloscopeai_pipeline` is a English model originally trained by celloscopeai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/celloscope_ner_banglabert_finetuned_celloscopeai_pipeline_en_5.5.1_3.0_1743744793480.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/celloscope_ner_banglabert_finetuned_celloscopeai_pipeline_en_5.5.1_3.0_1743744793480.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("celloscope_ner_banglabert_finetuned_celloscopeai_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("celloscope_ner_banglabert_finetuned_celloscopeai_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|celloscope_ner_banglabert_finetuned_celloscopeai_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|412.1 MB|

## References

https://huggingface.co/celloscopeai/celloscope-ner-banglabert-finetuned

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification