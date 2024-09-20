---
layout: model
title: English bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline pipeline BertForQuestionAnswering from rsml
author: John Snow Labs
name: bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline
date: 2024-09-19
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline` is a English model originally trained by rsml.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline_en_5.5.0_3.0_1726765316323.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline_en_5.5.0_3.0_1726765316323.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bluebert_pubmed_mimic_uncased_l_12_h_768_a_12_finetuned_squad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.1 MB|

## References

https://huggingface.co/rsml/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12-finetuned-squad

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering