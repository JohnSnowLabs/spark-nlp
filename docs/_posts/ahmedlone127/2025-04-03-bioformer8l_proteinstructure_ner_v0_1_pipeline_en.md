---
layout: model
title: English bioformer8l_proteinstructure_ner_v0_1_pipeline pipeline BertForTokenClassification from PDBEurope
author: John Snow Labs
name: bioformer8l_proteinstructure_ner_v0_1_pipeline
date: 2025-04-03
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bioformer8l_proteinstructure_ner_v0_1_pipeline` is a English model originally trained by PDBEurope.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bioformer8l_proteinstructure_ner_v0_1_pipeline_en_5.5.1_3.0_1743710700081.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bioformer8l_proteinstructure_ner_v0_1_pipeline_en_5.5.1_3.0_1743710700081.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bioformer8l_proteinstructure_ner_v0_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bioformer8l_proteinstructure_ner_v0_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bioformer8l_proteinstructure_ner_v0_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|158.6 MB|

## References

https://huggingface.co/PDBEurope/Bioformer8L-ProteinStructure-NER-v0.1

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification