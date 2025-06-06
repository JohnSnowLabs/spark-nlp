---
layout: model
title: English mitre_v16_tactic_bert_case_based_pipeline pipeline BertForSequenceClassification from sarahwei
author: John Snow Labs
name: mitre_v16_tactic_bert_case_based_pipeline
date: 2025-04-05
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mitre_v16_tactic_bert_case_based_pipeline` is a English model originally trained by sarahwei.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mitre_v16_tactic_bert_case_based_pipeline_en_5.5.1_3.0_1743869383376.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mitre_v16_tactic_bert_case_based_pipeline_en_5.5.1_3.0_1743869383376.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mitre_v16_tactic_bert_case_based_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mitre_v16_tactic_bert_case_based_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mitre_v16_tactic_bert_case_based_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.2 MB|

## References

https://huggingface.co/sarahwei/MITRE-v16-tactic-bert-case-based

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification