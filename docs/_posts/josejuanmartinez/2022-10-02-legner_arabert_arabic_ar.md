---
layout: model
title: Extract Financial, Legal and Generic Entities in Arabic
author: John Snow Labs
name: legner_arabert_arabic
date: 2022-10-02
tags: [ar, legal, licensed]
task: Named Entity Recognition
language: ar
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

AraBert v2 model, trained in house, using the dataset available [here](https://ontology.birzeit.edu/Wojood/) and augmenting with financial and legal information.

The entities you can find in this model are:

PERS (person)	
EVENT	
CARDINAL
NORP (group of people)
DATE
ORDINAL
OCC (occupation)
TIME
PERCENT
ORG (organization)
LANGUAGE
QUANTITY
GPE (geopolitical entity)
WEBSITE
UNIT
LOC (geographical location)
LAW
MONEY
FAC (facility: landmarks places)
PRODUCT
CURR (currency)

## Predicted Entities

`NORP`, `PERS`, `LOC`, `MONEY`, `TIME`, `ORG`, `WEBSITE`, `ORDINAL`, `PERCENT`, `EVENT`, `QUANTITY`, `OCC`, `LANGUAGE`, `CARDINAL`, `DATE`, `GPE`, `PRODUCT`, `CURR`, `FAC`, `UNIT`, `LAW`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_arabert_arabic_ar_1.0.0_3.0_1664705605292.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_arabert_arabic_ar_1.0.0_3.0_1664705605292.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokeniz = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")
  
tokenClassifier = legal.BertForTokenClassification.pretrained("legner_arabert_arabic", "ar", "legal/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")
  
pipeline = Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier])

example = spark.createDataFrame(pd.DataFrame({'text': ["""أمثلة:
جامعة بيرزيت وبالتعاون مع مؤسسة ادوارد سعيد تنظم مهرجان للفن الشعبي سيبدأ الساعة الرابعة عصرا، بتاريخ 16/5/2016.
بورصة فلسطين تسجل ارتفاعا بنسبة 0.08% ، في جلسة بلغت قيمة تداولاتها أكثر من نصف مليون دولار .
إنتخاب رئيس هيئة سوق رأس المال وتعديل مادة (4) في القانون الأساسي.
مسيرة قرب باب العامود والذي 700 متر عن المسجد الأقصى."""]}))

result = pipeline.fit(example).transform(example)
```

</div>

## Results

```bash
["أمثلة:","O"]
["جامعة","B-ORG"]
["بيرزيت","I-ORG"]
["وبالتعاون","O"]
["مع","O"]
["مؤسسة","B-ORG"]
["ادوارد","B-PERS"]
["سعيد","I-PERS"]
["تنظم","O"]
["مهرجان","B-EVENT"]
["للفن","I-EVENT"]
["الشعبي","I-EVENT"]
["سيبدأ","O"]
["الساعة","B-TIME"]
["الرابعة","I-TIME"]
["عصرا،","I-TIME"]
["بتاريخ","B-DATE"]
["16/5/2016.","I-DATE"]
["بورصة","B-ORG"]
["فلسطين","I-ORG"]
["تسجل","O"]
["ارتفاعا","O"]
["بنسبة","O"]
["0.08%","B-PERCENT"]
["،","O"]
["في","O"]
["جلسة","O"]
["بلغت","O"]
["قيمة","O"]
["تداولاتها","O"]
["أكثر","O"]
["من","O"]
["نصف","B-MONEY"]
["مليون","I-MONEY"]
["دولار","B-CURR"]
[".","O"]
["إنتخاب","O"]
["رئيس","B-OCC"]
["هيئة","B-ORG"]
["سوق","I-ORG"]
["رأس","I-ORG"]
["المال","I-ORG"]
["وتعديل","O"]
["مادة","B-LAW"]
["(4)","I-LAW"]
["في","O"]
["القانون","B-LAW"]
["الأساسي.","O"]
["مسيرة","O"]
["قرب","O"]
["باب","B-FAC"]
["العامود","I-FAC"]
["والذي","O"]
["700","B-QUANTITY"]
["متر","B-UNIT"]
["عن","O"]
["المسجد","B-FAC"]
["الأقصى.","I-FAC"]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_arabert_arabic|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|ar|
|Size:|505.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

https://ontology.birzeit.edu/Wojood/

## Benchmarking

```bash
       label  precision    recall  f1-score   support
  B-CARDINAL       0.93      0.87      0.80        19
      B-DATE       0.88      0.93      0.90       106
     B-EVENT       1.00      0.86      0.92        14
       B-FAC       1.00      0.67      0.80         3
       B-GPE       0.88      0.85      0.87        89
       B-LAW       1.00      0.50      0.67         6
      B-NORP       0.72      0.81      0.76        32
       B-OCC       0.88      0.83      0.85        52
   B-ORDINAL       0.76      0.80      0.78        35
       B-ORG       0.81      0.87      0.84       103
      B-PERS       0.78      0.89      0.83        47
   B-WEBSITE       0.62      1.00      0.77         5
  I-CARDINAL       0.33      0.62      0.43         8
      I-DATE       0.98      0.99      0.98       447
     I-EVENT       0.91      0.91      0.91        23
       I-FAC       0.75      0.43      0.55         7
       I-GPE       0.80      0.92      0.86        53
       I-LAW       1.00      0.85      0.92        13
      I-NORP       0.65      0.48      0.55        23
       I-OCC       0.97      0.86      0.91        96
       I-ORG       0.87      0.91      0.89       139
      I-PERS       0.94      1.00      0.97        60
   I-WEBSITE       0.94      1.00      0.97        15
           O       0.98      0.97      0.98      3062
    accuracy         -         -       0.95      4468
   macro-avg       0.83      0.81      0.81      4468
weighted-avg       0.95      0.95      0.95      4468
```
