---
layout: model
title: Persian imvladikon_charbert_roberta_wiki_pipeline pipeline RoBertaForTokenClassification from PerSpaCor
author: John Snow Labs
name: imvladikon_charbert_roberta_wiki_pipeline
date: 2024-09-19
tags: [fa, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`imvladikon_charbert_roberta_wiki_pipeline` is a Persian model originally trained by PerSpaCor.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/imvladikon_charbert_roberta_wiki_pipeline_fa_5.5.0_3.0_1726731192195.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/imvladikon_charbert_roberta_wiki_pipeline_fa_5.5.0_3.0_1726731192195.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("imvladikon_charbert_roberta_wiki_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("imvladikon_charbert_roberta_wiki_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|imvladikon_charbert_roberta_wiki_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|466.2 MB|

## References

https://huggingface.co/PerSpaCor/imvladikon-charbert-roberta-wiki

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification