---
layout: model
title: English tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline pipeline BertForTokenClassification from muhtasham
author: John Snow Labs
name: tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline
date: 2025-03-28
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline` is a English model originally trained by muhtasham.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline_en_5.5.1_3.0_1743186698753.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline_en_5.5.1_3.0_1743186698753.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tiny_mlm_rotten_tomatoes_from_scratch_target_conll2003_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|16.5 MB|

## References

https://huggingface.co/muhtasham/tiny-mlm-rotten_tomatoes-from-scratch-target-conll2003

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification