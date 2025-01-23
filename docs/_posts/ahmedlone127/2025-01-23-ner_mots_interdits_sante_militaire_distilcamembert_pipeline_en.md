---
layout: model
title: English ner_mots_interdits_sante_militaire_distilcamembert_pipeline pipeline CamemBertForTokenClassification from Steve77
author: John Snow Labs
name: ner_mots_interdits_sante_militaire_distilcamembert_pipeline
date: 2025-01-23
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

Pretrained CamemBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ner_mots_interdits_sante_militaire_distilcamembert_pipeline` is a English model originally trained by Steve77.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_mots_interdits_sante_militaire_distilcamembert_pipeline_en_5.5.1_3.0_1737641626994.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_mots_interdits_sante_militaire_distilcamembert_pipeline_en_5.5.1_3.0_1737641626994.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ner_mots_interdits_sante_militaire_distilcamembert_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ner_mots_interdits_sante_militaire_distilcamembert_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_mots_interdits_sante_militaire_distilcamembert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|253.6 MB|

## References

https://huggingface.co/Steve77/ner-mots_interdits-sante_militaire-distilcamembert

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForTokenClassification