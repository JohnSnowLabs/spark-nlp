---
layout: model
title: Pipeline to Detect Chemical Compounds and Genes (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_chemprot_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, berfortokenclassification, chemprot, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ner_chemprot](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_chemprot_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemprot_pipeline_en_3.4.1_3.0_1647889733985.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemprot_pipeline_en_3.4.1_3.0_1647889733985.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_token_classifier_ner_chemprot_pipeline", "en", "clinical/models")


pipeline.annotate("Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.")
```
```scala
val pipeline = new PretrainedPipeline("bert_token_classifier_ner_chemprot_pipeline", "en", "clinical/models")


pipeline.annotate("Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.")
```
</div>

## Results

```bash
+-------------------------------+---------+
|chunk                          |ner_label|
+-------------------------------+---------+
|Keratinocyte growth factor     |GENE-Y   |
|acidic fibroblast growth factor|GENE-Y   |
+-------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_chemprot_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter