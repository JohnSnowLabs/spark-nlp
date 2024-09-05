---
layout: model
title: Danish bert_token_classifier_danish_botxo_ner_dane_pipeline pipeline BertForTokenClassification from Maltehb
author: John Snow Labs
name: bert_token_classifier_danish_botxo_ner_dane_pipeline
date: 2024-09-04
tags: [da, open_source, pipeline, onnx]
task: Named Entity Recognition
language: da
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_token_classifier_danish_botxo_ner_dane_pipeline` is a Danish model originally trained by Maltehb.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_danish_botxo_ner_dane_pipeline_da_5.5.0_3.0_1725450260285.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_danish_botxo_ner_dane_pipeline_da_5.5.0_3.0_1725450260285.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_token_classifier_danish_botxo_ner_dane_pipeline", lang = "da")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_token_classifier_danish_botxo_ner_dane_pipeline", lang = "da")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_danish_botxo_ner_dane_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|da|
|Size:|412.3 MB|

## References

https://huggingface.co/Maltehb/danish-bert-botxo-ner-dane

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification