---
layout: model
title: Emotional Stress Classifier (BERT)
author: John Snow Labs
name: bert_sequence_classifier_stress
date: 2022-06-28
tags: [sequence_classification, bert, en, licensed, stress, mental, public_health]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [PHS-BERT-based](https://huggingface.co/publichealthsurveillance/PHS-BERT) classifier that can classify whether the content of a text expresses emotional stress.

## Predicted Entities

`no stress`, `stress`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_STRESS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_stress_en_4.0.0_3.0_1656438010655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_stress", "en", "clinical/models")\
.setInputCols(["document","token"])\
.setOutputCol("class")

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

data = spark.createDataFrame([["No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols("sentences")
.setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_stress", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))


val data = Seq("No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.stress").predict("""No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?""")
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
|text                                                                                                                                                                  |   class|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
|No place in my city has shelter space for us, and I won't put my baby on the literal street. What cities have good shelter programs for homeless mothers and children?|[stress]|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_stress|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

[Dreaddit dataset](https://arxiv.org/abs/1911.00133)

## Benchmarking

```bash
label           precision  recall    f1-score    support    
no-stress       0.83       0.82      0.83        334
stress          0.85       0.85      0.85        377
accuracy          -          -       0.84        711
macro-avg       0.84       0.84      0.84        711
weighted-avg    0.84       0.84      0.84        711
```