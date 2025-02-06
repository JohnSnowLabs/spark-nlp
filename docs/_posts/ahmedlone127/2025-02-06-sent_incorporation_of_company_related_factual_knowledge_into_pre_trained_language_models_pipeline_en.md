---
layout: model
title: English sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline pipeline BertSentenceEmbeddings from sophia-jihye
author: John Snow Labs
name: sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline
date: 2025-02-06
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

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline` is a English model originally trained by sophia-jihye.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline_en_5.5.1_3.0_1738840842913.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline_en_5.5.1_3.0_1738840842913.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_incorporation_of_company_related_factual_knowledge_into_pre_trained_language_models_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.0 MB|

## References

https://huggingface.co/sophia-jihye/Incorporation_of_Company-Related_Factual_Knowledge_into_Pre-trained_Language_Models

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings