---
layout: model
title: Pipeline to Detect Chemicals in Medical Texts
author: John Snow Labs
name: bert_token_classifier_ner_chemicals_pipeline
date: 2022-03-14
tags: [chemicals, bert_token_classifier, pipeline, ner, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ner_chemicals](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_chemicals_en.html) model.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CHEMICALS/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_pipeline_en_3.4.1_3.0_1647256416720.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_pipeline_en_3.4.1_3.0_1647256416720.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
chemicals_pipeline = PretrainedPipeline("bert_token_classifier_ner_chemicals_pipeline", "en", "clinical/models")

chemicals_pipeline.annotate("""The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.""")
```
```scala
val chemicals_pipeline = new PretrainedPipeline("bert_token_classifier_ner_chemicals_pipeline", "en", "clinical/models")

chemicals_pipeline.annotate("The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.")
```
</div>

## Results

```bash
+---------------------------+---------+
|chunk                      |ner_label|
+---------------------------+---------+
|p - choloroaniline         |CHEM     |
|chlorhexidine - digluconate|CHEM     |
|kanamycin                  |CHEM     |
|colistin                   |CHEM     |
|povidone - iodine          |CHEM     |
+---------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_chemicals_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.3 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter
- Finisher
