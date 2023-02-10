---
layout: model
title: Gender Classifier (BERT)
author: John Snow Labs
name: bert_sequence_classifier_gender_biobert
date: 2022-02-08
tags: [bert, sequence_classification, en, licensed]
task: Text Classification
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model classifies the gender of a patient in a clinical document using context.


This model is a [BioBERT-based](https://github.com/dmis-lab/biobert) classifier.


## Predicted Entities


`Female`, `Male`, `Unknown`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_gender_biobert_en_3.4.1_3.0_1644317917385.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_gender_biobert_en_3.4.1_3.0_1644317917385.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_gender_biobert", "en", "clinical/models")\
.setInputCols(["document","token"]) \
.setOutputCol("class") \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)


pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])


data = spark.createDataFrame([["The patient took Advil and he experienced an adverse reaction."]]).toDF("text")


result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")


val tokenizer = new Tokenizer()
.setInputCols("sentences")
.setOutputCol("token")


val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_gender_biobert", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("class")


val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))


val data = Seq("The patient took Advil and he experienced an adverse reaction.").toDF("text")


val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.gender.seq_biobert").predict("""The patient took Advil and he experienced an adverse reaction.""")
```

</div>


## Results


```bash
+---------------------------------------------------------------+------+
|text                                                           |result|
+---------------------------------------------------------------+------+
|The patient took Advil and he experienced an adverse reaction. |[Male]|
+---------------------------------------------------------------+------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_gender_biobert|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|


## References


This model is trained on more than four thousands clinical documents (radiology reports, pathology reports, clinical visits, etc) annotated internally.


## Benchmarking


```bash
label  precision  recall  f1-score  support
Female       0.94    0.94      0.94      479
Male       0.88    0.86      0.87      245
Unknown       0.73    0.78      0.76      102
accuracy       0.89    0.89      0.89      826
macro-avg       0.85    0.86      0.85      826
weighted-avg       0.90    0.89      0.90      826
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg3MDgwNzQxNCwzMzM0OTU5ODAsLTIwMj
UwMDIxMl19
-->