---
layout: model
title: Extract conditions and benefits from drug reviews
author: John Snow Labs
name: bert_token_classifier_ner_supplement
date: 2022-02-09
tags: [bertfortokenclassification, licensed, ner, en, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is trained to extract benefits of using drugs for certain conditions.


## Predicted Entities


`CONDITION`, `BENEFIT`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_supplement_en_3.0.2_3.0_1644368324280.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_supplement_en_3.0.2_3.0_1644368324280.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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


tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_supplement","en", "clinical/models")\
  .setInputCols(["token", "document"])\
  .setOutputCol("ner")\
  .setCaseSensitive(True)


ner_converter = NerConverter()\
        .setInputCols(["document","token","ner"])\
        .setOutputCol("ner_chunk")




pipeline =  Pipeline(
    stages=[
  documentAssembler,
  tokenizer,
  tokenClassifier,
  ner_converter
    ]
)


sample_df = spark.createDataFrame([["Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth. I recommend :)"],["Eager to have my ferritin grow and less hair loss."]]).toDF("text")


result = pipeline.fit(sample_df).transform(sample_df)
```
```scala
val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")


val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")


val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_supplement", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("ner")
    .setCaseSensitive(True)


val ner_converter = new NerConverter()
    .setInputCols(Array("document","token","ner"))
    .setOutputCol("ner_chunk")


val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))


val test_sentence = "Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth. I recommend :)"


val data = Seq(test_sentence).toDF(“text”) 


Val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
+-----------+---------+
|chunk      |ner_label|
+-----------+---------+
|nervousness|CONDITION|
|night sleep|BENEFIT  |
|hair       |BENEFIT  |
|nail growth|BENEFIT  |
|ferritin   |BENEFIT  |
|hair loss  |CONDITION|
+-----------+---------+


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_supplement|
|Compatibility:|Healthcare NLP 3.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|


## References


Trained on healthsea dataset: https://github.com/explosion/healthsea/tree/main/project/assets/ner


## Benchmarking


```bash
       label  precision  recall    f1  support
   B-BENEFIT       0.85    0.89  0.87      184
 B-CONDITION       0.82    0.90  0.86      202
   I-BENEFIT       0.83    0.70  0.76       64
 I-CONDITION       0.81    0.76  0.78      100
           O       1.00    0.99  1.00    12700
    accuracy       0.99    0.99  0.99    13250
   macro-avg       0.86    0.85  0.85    13250
weighted-avg       0.99    0.99  0.99    13250
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg4Mjk1NDMzMCwtNTc1ODA4NzA0XX0=
-->