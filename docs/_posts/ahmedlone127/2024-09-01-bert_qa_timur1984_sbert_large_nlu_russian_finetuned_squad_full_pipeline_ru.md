---
layout: model
title: Russian bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline pipeline BertForQuestionAnswering from Timur1984
author: John Snow Labs
name: bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline
date: 2024-09-01
tags: [ru, open_source, pipeline, onnx]
task: Question Answering
language: ru
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline` is a Russian model originally trained by Timur1984.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline_ru_5.4.2_3.0_1725151562647.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline_ru_5.4.2_3.0_1725151562647.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_qa_timur1984_sbert_large_nlu_russian_finetuned_squad_full_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|1.6 GB|

## References

https://huggingface.co/Timur1984/sbert_large_nlu_ru-finetuned-squad-full

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering