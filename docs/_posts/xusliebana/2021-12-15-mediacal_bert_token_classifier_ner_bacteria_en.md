---
layout: model
title: Medical Detect Bacterial Species (BertForTokenClassification)
author: John Snow Labs
name: mediacal_bert_token_classifier_ner_bacteria
date: 2021-12-15
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect different types of species of bacteria in text using pretrained NER model. This model is trained with the BertForTokenClassification method from transformers library and imported into Spark NLP.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/mediacal_bert_token_classifier_ner_bacteria_en_3.4.0_3.0_1639590602817.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

tokenClassifier =  MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bacteria", "en", "clinical/models")\
  .setInputCols("token", "document")\
  .setOutputCol("ner")\
  .setCaseSensitive(True)

ner_converter = NerConverter()\
        .setInputCols(["document","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents \
a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica \
sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T))."""

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
```scala
val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_bacteria", "en", "clincal/models")
  .setInputCols("token", "document")
  .setOutputCol("ner")
  .setCaseSensitive(True)

val ner_converter = NerConverter()
        .setInputCols(Array("document","token","ner"))
        .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|SMSP (T)               |SPECIES  |
|Methanoregula formicica|SPECIES  |
|SMSP (T)               |SPECIES  |
+-----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mediacal_bert_token_classifier_ner_bacteria|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.1 MB|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

Trained on a custom dataset by John Snow Labs.

## Benchmarking

```bash
                         precision    recall  f1-score   support

B-SPECIES       0.98           0.84     0.91        767
I-SPECIES        0.99           0.84     0.91        1043
accuracy                                         0.84        1810
macro avg       0.85          0.89      0.87        1810
weighted avg  0.99          0.84      0.91        1810
```