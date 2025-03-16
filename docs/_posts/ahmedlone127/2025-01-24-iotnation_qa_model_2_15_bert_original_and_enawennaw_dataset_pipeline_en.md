---
layout: model
title: English iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline pipeline BertForQuestionAnswering from chriskim2273
author: John Snow Labs
name: iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline
date: 2025-01-24
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline` is a English model originally trained by chriskim2273.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline_en_5.5.1_3.0_1737752090153.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline_en_5.5.1_3.0_1737752090153.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|iotnation_qa_model_2_15_bert_original_and_enawennaw_dataset_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|403.7 MB|

## References

https://huggingface.co/chriskim2273/IOTNation_QA_Model_2.15_BERT_ORIGINAL_AND_UNK_DATASET

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering