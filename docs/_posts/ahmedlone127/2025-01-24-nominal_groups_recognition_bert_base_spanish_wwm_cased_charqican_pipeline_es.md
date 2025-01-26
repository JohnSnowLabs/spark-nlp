---
layout: model
title: Castilian, Spanish nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline pipeline BertForTokenClassification from charqican
author: John Snow Labs
name: nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline
date: 2025-01-24
tags: [es, open_source, pipeline, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline` is a Castilian, Spanish model originally trained by charqican.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline_es_5.5.1_3.0_1737719739509.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline_es_5.5.1_3.0_1737719739509.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nominal_groups_recognition_bert_base_spanish_wwm_cased_charqican_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|409.5 MB|

## References

https://huggingface.co/charqican/nominal-groups-recognition-bert-base-spanish-wwm-cased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification