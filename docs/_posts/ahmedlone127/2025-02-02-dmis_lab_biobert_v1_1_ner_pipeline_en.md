---
layout: model
title: English dmis_lab_biobert_v1_1_ner_pipeline pipeline BertForTokenClassification from MDDDDR
author: John Snow Labs
name: dmis_lab_biobert_v1_1_ner_pipeline
date: 2025-02-02
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dmis_lab_biobert_v1_1_ner_pipeline` is a English model originally trained by MDDDDR.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dmis_lab_biobert_v1_1_ner_pipeline_en_5.5.1_3.0_1738469566387.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dmis_lab_biobert_v1_1_ner_pipeline_en_5.5.1_3.0_1738469566387.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dmis_lab_biobert_v1_1_ner_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dmis_lab_biobert_v1_1_ner_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dmis_lab_biobert_v1_1_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.1 MB|

## References

https://huggingface.co/MDDDDR/dmis_lab_biobert_v1.1_NER

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification