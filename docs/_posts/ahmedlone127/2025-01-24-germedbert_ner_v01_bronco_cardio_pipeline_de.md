---
layout: model
title: German germedbert_ner_v01_bronco_cardio_pipeline pipeline BertForTokenClassification from BachelorThesis
author: John Snow Labs
name: germedbert_ner_v01_bronco_cardio_pipeline
date: 2025-01-24
tags: [de, open_source, pipeline, onnx]
task: Named Entity Recognition
language: de
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`germedbert_ner_v01_bronco_cardio_pipeline` is a German model originally trained by BachelorThesis.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/germedbert_ner_v01_bronco_cardio_pipeline_de_5.5.1_3.0_1737719896815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/germedbert_ner_v01_bronco_cardio_pipeline_de_5.5.1_3.0_1737719896815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("germedbert_ner_v01_bronco_cardio_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("germedbert_ner_v01_bronco_cardio_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|germedbert_ner_v01_bronco_cardio_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|406.9 MB|

## References

https://huggingface.co/BachelorThesis/GerMedBERT_NER_V01_BRONCO_CARDIO

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification