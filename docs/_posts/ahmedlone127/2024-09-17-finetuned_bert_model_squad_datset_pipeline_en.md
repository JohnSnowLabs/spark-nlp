---
layout: model
title: English finetuned_bert_model_squad_datset_pipeline pipeline DistilBertForQuestionAnswering from AlyGreo
author: John Snow Labs
name: finetuned_bert_model_squad_datset_pipeline
date: 2024-09-17
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

Pretrained DistilBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_bert_model_squad_datset_pipeline` is a English model originally trained by AlyGreo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_bert_model_squad_datset_pipeline_en_5.5.0_3.0_1726555344077.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_bert_model_squad_datset_pipeline_en_5.5.0_3.0_1726555344077.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_bert_model_squad_datset_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_bert_model_squad_datset_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_bert_model_squad_datset_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/AlyGreo/finetuned-bert-model-squad-datset

## Included Models

- MultiDocumentAssembler
- DistilBertForQuestionAnswering