---
layout: model
title: English sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline pipeline BertSentenceEmbeddings from nreimers
author: John Snow Labs
name: sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline
date: 2025-04-01
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline` is a English model originally trained by nreimers.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline_en_5.5.1_3.0_1743506103672.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline_en_5.5.1_3.0_1743506103672.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_minilmv2_l6_h384_distilled_from_bert_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|54.7 MB|

## References

References

https://huggingface.co/nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Base

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings