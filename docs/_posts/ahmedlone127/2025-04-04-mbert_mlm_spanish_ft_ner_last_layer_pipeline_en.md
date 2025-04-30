---
layout: model
title: English mbert_mlm_spanish_ft_ner_last_layer_pipeline pipeline BertForTokenClassification from ashwathjadhav23
author: John Snow Labs
name: mbert_mlm_spanish_ft_ner_last_layer_pipeline
date: 2025-04-04
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mbert_mlm_spanish_ft_ner_last_layer_pipeline` is a English model originally trained by ashwathjadhav23.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mbert_mlm_spanish_ft_ner_last_layer_pipeline_en_5.5.1_3.0_1743745784455.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mbert_mlm_spanish_ft_ner_last_layer_pipeline_en_5.5.1_3.0_1743745784455.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mbert_mlm_spanish_ft_ner_last_layer_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mbert_mlm_spanish_ft_ner_last_layer_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mbert_mlm_spanish_ft_ner_last_layer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|665.1 MB|

## References

https://huggingface.co/ashwathjadhav23/mbert_mlm_es_ft_ner_last_layer

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification