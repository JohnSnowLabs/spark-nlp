---
layout: model
title: Indonesian xlmroberta_ner_base_indonesian_pipeline pipeline XlmRoBertaForTokenClassification from cahya
author: John Snow Labs
name: xlmroberta_ner_base_indonesian_pipeline
date: 2024-06-10
tags: [id, open_source, pipeline, onnx]
task: Named Entity Recognition
language: id
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_ner_base_indonesian_pipeline` is a Indonesian model originally trained by cahya.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_indonesian_pipeline_id_5.4.0_3.0_1718023514998.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_indonesian_pipeline_id_5.4.0_3.0_1718023514998.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_ner_base_indonesian_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_ner_base_indonesian_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_base_indonesian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|791.6 MB|

## References

https://huggingface.co/cahya/xlm-roberta-base-indonesian-NER

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification