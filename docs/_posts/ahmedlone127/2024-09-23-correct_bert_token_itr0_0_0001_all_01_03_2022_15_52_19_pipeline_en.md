---
layout: model
title: English correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline pipeline BertForTokenClassification from ali2066
author: John Snow Labs
name: correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline
date: 2024-09-23
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline` is a English model originally trained by ali2066.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline_en_5.5.0_3.0_1727111290012.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline_en_5.5.0_3.0_1727111290012.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|correct_bert_token_itr0_0_0001_all_01_03_2022_15_52_19_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/ali2066/correct_BERT_token_itr0_0.0001_all_01_03_2022-15_52_19

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification