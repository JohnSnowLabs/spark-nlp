---
layout: model
title: English distilbert_uncased_ner_lora_chidamnat2002_pipeline pipeline DistilBertForTokenClassification from chidamnat2002
author: John Snow Labs
name: distilbert_uncased_ner_lora_chidamnat2002_pipeline
date: 2024-11-11
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

Pretrained DistilBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`distilbert_uncased_ner_lora_chidamnat2002_pipeline` is a English model originally trained by chidamnat2002.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_uncased_ner_lora_chidamnat2002_pipeline_en_5.5.1_3.0_1731327345399.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_uncased_ner_lora_chidamnat2002_pipeline_en_5.5.1_3.0_1731327345399.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("distilbert_uncased_ner_lora_chidamnat2002_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("distilbert_uncased_ner_lora_chidamnat2002_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_uncased_ner_lora_chidamnat2002_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/chidamnat2002/distilbert-uncased-NER-LoRA

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification