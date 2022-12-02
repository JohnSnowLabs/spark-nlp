---
layout: model
title: NER Pipeline for 6 Scandinavian Languages
author: John Snow Labs
name: bert_token_classifier_scandi_ner_pipeline
date: 2022-02-15
tags: [danish, norwegian, swedish, icelandic, faroese, bert, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on [bert_token_classifier_scandi_ner](https://nlp.johnsnowlabs.com/2021/12/09/bert_token_classifier_scandi_ner_xx.html) model which is imported from `HuggingFace`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_SCANDINAVIAN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_scandi_ner_pipeline_xx_3.4.0_3.0_1644927539839.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
+-------------------+---------+
|chunk              |ner_label|
+-------------------+---------+
|Hans               |PER      |
|Statens Universitet|ORG      |
|København          |LOC      |
|københavner        |MISC     |
+-------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_scandi_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.0+|
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