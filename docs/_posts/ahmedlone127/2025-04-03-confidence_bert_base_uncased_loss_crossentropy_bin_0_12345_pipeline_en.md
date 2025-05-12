---
layout: model
title: English confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline pipeline BertForSequenceClassification from kevinr
author: John Snow Labs
name: confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline
date: 2025-04-03
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline` is a English model originally trained by kevinr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline_en_5.5.1_3.0_1743679314961.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline_en_5.5.1_3.0_1743679314961.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|confidence_bert_base_uncased_loss_crossentropy_bin_0_12345_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.4 MB|

## References

https://huggingface.co/kevinr/Confidence-bert-base-uncased-Loss_CrossEntropy-Bin_0-12345

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification