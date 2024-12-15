---
layout: model
title: English pubmed_bert_mlm_squad_covidqa_pipeline pipeline BertForQuestionAnswering from Sarmila
author: John Snow Labs
name: pubmed_bert_mlm_squad_covidqa_pipeline
date: 2024-12-15
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pubmed_bert_mlm_squad_covidqa_pipeline` is a English model originally trained by Sarmila.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pubmed_bert_mlm_squad_covidqa_pipeline_en_5.5.1_3.0_1734297575356.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pubmed_bert_mlm_squad_covidqa_pipeline_en_5.5.1_3.0_1734297575356.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("pubmed_bert_mlm_squad_covidqa_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("pubmed_bert_mlm_squad_covidqa_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pubmed_bert_mlm_squad_covidqa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.1 MB|

## References

https://huggingface.co/Sarmila/pubmed-bert-mlm-squad-covidqa

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering