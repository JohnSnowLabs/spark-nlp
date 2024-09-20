---
layout: model
title: Multilingual bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline pipeline BertForTokenClassification from StivenLancheros
author: John Snow Labs
name: bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline
date: 2024-09-05
tags: [xx, open_source, pipeline, onnx]
task: Named Entity Recognition
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline` is a Multilingual model originally trained by StivenLancheros.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline_xx_5.5.0_3.0_1725511073548.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline_xx_5.5.0_3.0_1725511073548.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_ner_biobert_base_cased_v1.2_finetuned_ner_craft_augmentedtransfer_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|403.7 MB|

## References

https://huggingface.co/StivenLancheros/biobert-base-cased-v1.2-finetuned-ner-CRAFT_AugmentedTransfer_EN

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification