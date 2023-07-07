---
layout: model
title: NER Pipeline for 6 Scandinavian Languages
author: John Snow Labs
name: bert_token_classifier_scandi_ner_pipeline
date: 2023-05-26
tags: [danish, norwegian, swedish, icelandic, faroese, bert, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on [bert_token_classifier_scandi_ner](https://nlp.johnsnowlabs.com/2021/12/09/bert_token_classifier_scandi_ner_xx.html) model which is imported from `HuggingFace`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_scandi_ner_pipeline_xx_4.4.2_3.4_1685062102199.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_scandi_ner_pipeline_xx_4.4.2_3.4_1685062102199.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
scandiner_pipeline = PretrainedPipeline("bert_token_classifier_scandi_ner_pipeline", lang = "xx")
scandiner_pipeline.annotate("Hans er professor ved Statens Universitet, som ligger i København, og han er en rigtig københavner.")
```
```scala
val scandiner_pipeline = new PretrainedPipeline("bert_token_classifier_scandi_ner_pipeline", lang = "xx")

val scandiner_pipeline.annotate("Hans er professor ved Statens Universitet, som ligger i København, og han er en rigtig københavner.")
```
</div>

## Results

```bash
Results



+-------------------+---------+
|chunk              |ner_label|
+-------------------+---------+
|Hans               |PER      |
|Statens Universitet|ORG      |
|København          |LOC      |
|københavner        |MISC     |
+-------------------+---------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_scandi_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|666.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- BertForTokenClassification
- NerConverter
- Finisher