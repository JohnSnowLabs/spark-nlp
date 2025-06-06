---
layout: model
title: English biolinkbert_base_ner_bc5cdr_d_zs_pipeline pipeline BertForTokenClassification from 68611-llm-annotation-group
author: John Snow Labs
name: biolinkbert_base_ner_bc5cdr_d_zs_pipeline
date: 2025-03-31
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`biolinkbert_base_ner_bc5cdr_d_zs_pipeline` is a English model originally trained by 68611-llm-annotation-group.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biolinkbert_base_ner_bc5cdr_d_zs_pipeline_en_5.5.1_3.0_1743391899715.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biolinkbert_base_ner_bc5cdr_d_zs_pipeline_en_5.5.1_3.0_1743391899715.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("biolinkbert_base_ner_bc5cdr_d_zs_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("biolinkbert_base_ner_bc5cdr_d_zs_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biolinkbert_base_ner_bc5cdr_d_zs_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.5 MB|

## References

https://huggingface.co/68611-llm-annotation-group/BioLinkBERT-base-NER-BC5CDR_D-ZS

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification