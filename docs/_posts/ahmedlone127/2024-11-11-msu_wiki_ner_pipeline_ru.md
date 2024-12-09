---
layout: model
title: Russian msu_wiki_ner_pipeline pipeline BertForTokenClassification from nesemenpolkov
author: John Snow Labs
name: msu_wiki_ner_pipeline
date: 2024-11-11
tags: [ru, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ru
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`msu_wiki_ner_pipeline` is a Russian model originally trained by nesemenpolkov.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/msu_wiki_ner_pipeline_ru_5.5.1_3.0_1731298735156.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/msu_wiki_ner_pipeline_ru_5.5.1_3.0_1731298735156.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("msu_wiki_ner_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("msu_wiki_ner_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|msu_wiki_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|665.1 MB|

## References

https://huggingface.co/nesemenpolkov/msu-wiki-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification