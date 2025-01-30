---
layout: model
title: English google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline pipeline BertForTokenClassification from Kudod
author: John Snow Labs
name: google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline
date: 2025-01-25
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline` is a English model originally trained by Kudod.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline_en_5.5.1_3.0_1737845147381.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline_en_5.5.1_3.0_1737845147381.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|google_bert_large_cased_finetuned_ner_vlsp2021_3090_15june_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/Kudod/google-bert-large-cased-finetuned-ner-vlsp2021-3090-15June-1

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification