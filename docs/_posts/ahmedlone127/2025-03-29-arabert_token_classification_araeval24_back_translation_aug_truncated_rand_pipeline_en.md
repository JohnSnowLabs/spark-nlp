---
layout: model
title: English arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline pipeline BertForTokenClassification from MM2157
author: John Snow Labs
name: arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline
date: 2025-03-29
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline` is a English model originally trained by MM2157.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline_en_5.5.1_3.0_1743219136424.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline_en_5.5.1_3.0_1743219136424.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arabert_token_classification_araeval24_back_translation_aug_truncated_rand_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|504.7 MB|

## References

https://huggingface.co/MM2157/AraBERT_token_classification_AraEval24_back_translation_aug_truncated_rand

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification