---
layout: model
title: English arbovirus_bert_base_lr_1e_5_n_5_pipeline pipeline BertEmbeddings from mmcleige
author: John Snow Labs
name: arbovirus_bert_base_lr_1e_5_n_5_pipeline
date: 2025-03-31
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arbovirus_bert_base_lr_1e_5_n_5_pipeline` is a English model originally trained by mmcleige.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arbovirus_bert_base_lr_1e_5_n_5_pipeline_en_5.5.1_3.0_1743453213256.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arbovirus_bert_base_lr_1e_5_n_5_pipeline_en_5.5.1_3.0_1743453213256.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arbovirus_bert_base_lr_1e_5_n_5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arbovirus_bert_base_lr_1e_5_n_5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arbovirus_bert_base_lr_1e_5_n_5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.1 MB|

## References

https://huggingface.co/mmcleige/arbovirus_bert_base_LR.1e-5_N.5

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings