---
layout: model
title: Detect Chemical Compounds and Genes (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_chemprot
date: 2022-01-06
tags: [berfortokenclassification, chemprot, licensed, en]
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


Detect chemical compounds and genes in the medical text using the pretrained NER model. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.


## Predicted Entities


`CHEMICAL`, `GENE-Y`, `GENE-N`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemprot_en_3.3.4_2.4_1641471274375.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemprot_en_3.3.4_2.4_1641471274375.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_chemprot", "en", "clinical/models")\
.setInputCols("token", "document")\
.setOutputCol("ner")\
.setCaseSensitive(True)

ner_converter = NerConverter()\
.setInputCols(["document","token","ner"])\
.setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

data = spark.createDataFrame([["""Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium."""
]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_chemprot", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("ner")
.setCaseSensitive(True)

val ner_converter = new NerConverter()
.setInputCols(Array("document","token","ner"))
.setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.chemprot.bert").predict("""Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.""")
```

</div>


## Results


```bash
+-------------------------------+---------+
|chunk                          |ner_label|
+-------------------------------+---------+
|Keratinocyte growth factor     |GENE-Y   |
|acidic fibroblast growth factor|GENE-Y   |
+-------------------------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_chemprot|
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


This model is trained on a [ChemProt corpus](https://biocreative.bioinformatics.udel.edu/).


## Benchmarking


```bash
label  precision    recall  f1-score   support
B-CHEMICAL       0.93      0.79      0.85      8649
B-GENE-N       0.63      0.56      0.59      2752
B-GENE-Y       0.82      0.73      0.77      5490
I-CHEMICAL       0.90      0.79      0.84      1313
I-GENE-N       0.72      0.62      0.67      1993
I-GENE-Y       0.81      0.72      0.77      2420
accuracy       -         -         0.73     22617
macro-avg       0.75      0.74      0.75     22617
weighted-avg       0.83      0.73      0.78     22617
```
