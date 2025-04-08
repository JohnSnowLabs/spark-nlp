---
layout: model
title: English original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline pipeline BertForTokenClassification from ghadeermobasher
author: John Snow Labs
name: original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline
date: 2025-04-08
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline` is a English model originally trained by ghadeermobasher.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline_en_5.5.1_3.0_1744090471830.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline_en_5.5.1_3.0_1744090471830.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|original_biomednlp_pubmedbert_base_uncased_abstract_biored_dis_256_16_5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.1 MB|

## References

https://huggingface.co/ghadeermobasher/Original-BiomedNLP-PubMedBERT-base-uncased-abstract-BioRED_Dis-256-16-5

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification