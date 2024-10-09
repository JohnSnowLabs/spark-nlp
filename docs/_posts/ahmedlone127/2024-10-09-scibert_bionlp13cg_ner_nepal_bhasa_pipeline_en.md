---
layout: model
title: English scibert_bionlp13cg_ner_nepal_bhasa_pipeline pipeline BertForTokenClassification from judithrosell
author: John Snow Labs
name: scibert_bionlp13cg_ner_nepal_bhasa_pipeline
date: 2024-10-09
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`scibert_bionlp13cg_ner_nepal_bhasa_pipeline` is a English model originally trained by judithrosell.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/scibert_bionlp13cg_ner_nepal_bhasa_pipeline_en_5.5.1_3.0_1728473865730.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/scibert_bionlp13cg_ner_nepal_bhasa_pipeline_en_5.5.1_3.0_1728473865730.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("scibert_bionlp13cg_ner_nepal_bhasa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("scibert_bionlp13cg_ner_nepal_bhasa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|scibert_bionlp13cg_ner_nepal_bhasa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|410.1 MB|

## References

https://huggingface.co/judithrosell/SciBERT_BioNLP13CG_NER_new

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification