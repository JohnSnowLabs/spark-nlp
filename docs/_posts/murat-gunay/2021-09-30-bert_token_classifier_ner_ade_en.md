---
layout: model
title: Detect Adverse Drug Events (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_ade
date: 2021-09-30
tags: [adverse, ade, bertfortokenclassification, ner, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.2.2
spark_version: 2.4
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Detect adverse reactions of drugs in reviews, tweets, and medical text using the pretrained NER model. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.


## Predicted Entities


`DRUG`, `ADE`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_BERT_TOKEN_CLASSIFIER/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_ade_en_3.2.2_2.4_1633008677011.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol('text')\
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_ade", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["document","token","ner"])\
    .setOutputCol("ner_chunk")


pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """Been taking Lipitor for 15 years , have experienced severe fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps"""

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_ade", "en", "clinical/models")
    .setInputCols(Array("token", "document"))
    .setOutputCol("ner")
    .setCaseSensitive(True)

val ner_converter = new NerConverter()
    .setInputCols(Array("document","token","ner"))
    .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}

```python
import nlu
nlu.load("en.classify.token_bert.ner_ade").predict("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""")
```

</div>


## Results


```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|Lipitor       |DRUG     |
|severe fatigue|ADE      |
|voltaren      |DRUG     |
|cramps        |ADE      |
+--------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_ade|
|Compatibility:|Healthcare NLP 3.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|


## Data Source


This model is trained on a custom dataset by John Snow Labs.


## Benchmarking


```bash
label         precision  recall  f1-score   support
B-ADE           0.93      0.79      0.85      2694
B-DRUG          0.97      0.87      0.92      9539
I-ADE           0.93      0.73      0.82      3236
I-DRUG          0.95      0.82      0.88      6115
accuracy        -         -         0.83     21584
macro-avg       0.84      0.84      0.84     21584
weighted-avg    0.95      0.83      0.89     21584
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU4NzIyMDQyOF19
-->