---
layout: model
title: English classif_mmate_1_5_original_cont_3_sent_pipeline pipeline BertForSequenceClassification from spneshaei
author: John Snow Labs
name: classif_mmate_1_5_original_cont_3_sent_pipeline
date: 2024-09-20
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`classif_mmate_1_5_original_cont_3_sent_pipeline` is a English model originally trained by spneshaei.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classif_mmate_1_5_original_cont_3_sent_pipeline_en_5.5.0_3.0_1726860348492.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classif_mmate_1_5_original_cont_3_sent_pipeline_en_5.5.0_3.0_1726860348492.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("classif_mmate_1_5_original_cont_3_sent_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("classif_mmate_1_5_original_cont_3_sent_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classif_mmate_1_5_original_cont_3_sent_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|409.1 MB|

## References

https://huggingface.co/spneshaei/classif_mmate_1_5_original_cont_3_sent

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification