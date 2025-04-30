---
layout: model
title: Serbian bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline pipeline BertForTokenClassification from ICEF-NLP
author: John Snow Labs
name: bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline
date: 2025-03-28
tags: [sr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: sr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline` is a Serbian model originally trained by ICEF-NLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline_sr_5.5.1_3.0_1743137491367.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline_sr_5.5.1_3.0_1743137491367.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline", lang = "sr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline", lang = "sr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bcms_bertic_comtext_serbian_legal_ner_ekavica_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|sr|
|Size:|412.8 MB|

## References

https://huggingface.co/ICEF-NLP/bcms-bertic-comtext-sr-legal-ner-ekavica

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification