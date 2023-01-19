---
layout: model
title: Detect Chemicals in Medical text (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_chemicals
date: 2021-10-19
tags: [berfortokenclassification, ner, chemicals, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.0
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
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_BERT_TOKEN_CLASSIFIER/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_en_3.3.0_2.4_1634649785035.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemicals_en_3.3.0_2.4_1634649785035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_chemicals", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)

ner_converter = NerConverter()\
    .setInputCols(["document","token","ner"])\
    .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. "A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis."""

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_chemicals", "en", "clinical/models")
    .setInputCols(Array("token", "document"))
    .setOutputCol("ner")
    .setCaseSensitive(True)

val ner_converter = new NerConverter()
    .setInputCols(Array("document","token","ner"))
    .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("The results have shown that the product p - choloroaniline is not a significant factor in chlorhexidine - digluconate associated erosive cystitis. "A high percentage of kanamycin - colistin and povidone - iodine irrigations were associated with erosive cystitis.").toDS.toDF("text")

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
|Compatibility:|Healthcare NLP 3.3.0+|
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
label           precision   recall   f1-score  support
B-CHEM             0.99      0.92      0.95     30731
I-CHEM             0.99      0.93      0.96     31270
accuracy           -         -         0.93     62001
macro-avg          0.96      0.95      0.96     62001
weighted-avg       0.99      0.93      0.96     62001
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNTc3NTE3MDNdfQ==
-->