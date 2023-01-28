---
layout: model
title: Detect Chemicals in Medical text (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_chemicals
date: 2022-01-06
tags: [berfortokenclassification, ner, chemicals, en, licensed]
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


Extract different types of chemical compounds mentioned in text using pretrained NER model. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.


## Predicted Entities


`CHEM`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_en_3.3.4_2.4_1641465134046.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_en_3.3.4_2.4_1641465134046.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_chemicals", "en", "clinical/models")\
.setInputCols("token", "document")\
.setOutputCol("ner")\
.setCaseSensitive(True)

ner_converter = NerConverter()\
.setInputCols(["document","token","ner"])\
.setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

data = spark.createDataFrame([["""The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. "A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_chemicals", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("ner")
.setCaseSensitive(True)

val ner_converter = NerConverter()
.setInputCols(Array("document","token","ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. "A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_chemical").predict("""The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. "A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.""")
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
|Model Name:|bert_token_classifier_ner_chemicals|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.3 MB|
|Case sensitive:|true|
|Max sentense length:|512|


## Data Source


This model is trained on a custom dataset by John Snow Labs.


## Benchmarking


```bash
label   precision    recall  f1-score  support
B-CHEM       0.99      0.92      0.95    30731
I-CHEM       0.99      0.93      0.96    31270
accuracy       -         -         0.93    62001
macro-avg       0.96      0.95      0.96    62001
weighted-avg       0.99      0.93      0.96    62001
```
