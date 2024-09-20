---
layout: model
title: English biomednlp_biomedbert_base_uncased_abstract_pipeline pipeline BertEmbeddings from microsoft
author: John Snow Labs
name: biomednlp_biomedbert_base_uncased_abstract_pipeline
date: 2024-09-03
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`biomednlp_biomedbert_base_uncased_abstract_pipeline` is a English model originally trained by microsoft.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biomednlp_biomedbert_base_uncased_abstract_pipeline_en_5.5.0_3.0_1725407061863.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biomednlp_biomedbert_base_uncased_abstract_pipeline_en_5.5.0_3.0_1725407061863.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("biomednlp_biomedbert_base_uncased_abstract_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("biomednlp_biomedbert_base_uncased_abstract_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biomednlp_biomedbert_base_uncased_abstract_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.1 MB|

## References

https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings