---
layout: model
title: Detect PHI for Deidentification (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_deid
date: 2022-01-06
tags: [licensed, berfortokenclassification, deid, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Deidentification NER is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 23 entities. This ner model is trained with a combination of the i2b2 train set and a re-augmented version of i2b2 train set using `BertForTokenClassification`


We sticked to official annotation guideline (AG) for 2014 i2b2 Deid challenge while annotating new datasets for this model. All the details regarding the nuances and explanations for AG can be found here [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4978170/)


## Predicted Entities


`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_deid_en_3.3.4_2.4_1641472006823.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

tokenizer = Tokenizer()\
.setInputCols(["document"])\
.setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_deid", "en", "clinical/models")\
.setInputCols(["token", "document"])\
.setOutputCol("ner")\
.setCaseSensitive(True)

ner_converter = NerConverter()\
.setInputCols(["document","token","ner"])\
.setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

data = spark.createDataFrame([["""A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine."""]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_deid", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("ner")
.setCaseSensitive(True)

val ner_converter = new NerConverter()
.setInputCols(Array("document","token","ner"))
.setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_deid").predict("""A. Record date : 2093-01-13, David Hale, M.D. Name : Hendrickson, Ora MR. # 7194334. PCP : Oliveira, non-smoking. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227. Patient's complaints first surfaced when he started working for Brothers Coal-Mine.""")
```

</div>


## Results


```bash
+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson, Ora             |PATIENT      |
|7194334                      |MEDICALRECORD|
|Oliveira                     |PATIENT      |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street            |STREET       |
|302) 786-5227                |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_deid|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.4 MB|
|Case sensitive:|true|
|Max sentense length:|128|


## Data Source


A custom data set which is created from the i2b2-PHI train and the re-augmented version of the i2b2-PHI train set is used.


## Benchmarking


```bash
label  precision    recall  f1-score   support
B-AGE       0.92      0.80      0.86      1050
B-CITY       0.71      0.93      0.80       530
B-COUNTRY       0.94      0.72      0.82       179
B-DATE       0.99      0.99      0.99     20434
B-DEVICE       0.68      0.66      0.67        35
B-DOCTOR       0.93      0.91      0.92      3609
B-EMAIL       0.92      1.00      0.96        11
B-HOSPITAL       0.94      0.90      0.92      2225
B-IDNUM       0.88      0.64      0.74      1185
B-LOCATION-OTHER       0.83      0.60      0.70        25
B-MEDICALRECORD       0.84      0.97      0.90      2263
B-ORGANIZATION       0.92      0.68      0.79       171
B-PATIENT       0.89      0.86      0.88      2297
B-PHONE       0.90      0.96      0.93       755
B-PROFESSION       0.86      0.81      0.83       265
B-STATE       0.97      0.87      0.92       261
B-STREET       0.98      0.99      0.99       184
B-USERNAME       0.96      0.78      0.86       357
B-ZIP       0.96      0.96      0.96       444
I-CITY       0.74      0.83      0.78       138
I-DATE       0.98      0.96      0.97       955
I-DEVICE       1.00      1.00      1.00         3
I-DOCTOR       0.92      0.98      0.95      3134
I-HOSPITAL       0.95      0.92      0.93      1322
I-LOCATION-OTHER       1.00      1.00      1.00         8
I-MEDICALRECORD       1.00      0.91      0.95        23
I-ORGANIZATION       0.98      0.61      0.75        77
I-PATIENT       0.89      0.81      0.85      1281
I-PHONE       0.97      0.96      0.97       374
I-PROFESSION       0.95      0.82      0.88       232
I-STREET       0.98      0.98      0.98       391
O       1.00      1.00      1.00    585606
accuracy       -         -         0.99    629960
macro-avg       0.79      0.71      0.73    629960
weighted-avg       0.99      0.99      0.99    629960
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI5NzYzNzUxMl19
-->