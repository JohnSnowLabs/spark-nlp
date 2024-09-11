---
layout: model
title: English hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline pipeline DistilBertEmbeddings from yuwei2342
author: John Snow Labs
name: hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline
date: 2024-09-09
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline` is a English model originally trained by yuwei2342.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline_en_5.5.0_3.0_1725905787309.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline_en_5.5.0_3.0_1725905787309.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hf_nlp_course_distilbert_base_uncased_finetuned_imdb_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.2 MB|

## References

https://huggingface.co/yuwei2342/hf-nlp-course-distilbert-base-uncased-finetuned-imdb

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings