---
layout: model
title: English cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline pipeline DistilBertForTokenClassification from Somisetty2347
author: John Snow Labs
name: cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline
date: 2025-01-29
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline` is a English model originally trained by Somisetty2347.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline_en_5.5.1_3.0_1738170882460.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline_en_5.5.1_3.0_1738170882460.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cleanedsomisetty_iucn_acc_num_pass_file2347_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|244.0 MB|

## References

https://huggingface.co/Somisetty2347/CLEANEDSOMISETTY_IUCN_ACC_NUM_PASS_file2347

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification