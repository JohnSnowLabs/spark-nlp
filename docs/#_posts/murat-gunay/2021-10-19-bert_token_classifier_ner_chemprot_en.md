---
layout: model
title: Detect Chemical Compounds and Genes (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_chemprot
date: 2021-10-19
tags: [berfortokenclassification, ner, chemprot, en, licensed]
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


Detect chemical compounds and genes in the medical text using the pretrained NER model. This model is trained with the `BertForTokenClassification` method from the `transformers` library and imported into Spark NLP.


## Predicted Entities


`CHEMICAL`, `GENE-Y`, `GENE-N`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CHEMPROT_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_CHEMPROT_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_chemprot_en_3.3.0_2.4_1634644903577.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
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

pipeline =  Pipeline(stages=[document_assembler, 
                             tokenizer, 
                             tokenClassifier, 
                             ner_converter])

sample_text = "Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium."

df = spark.createDataFrame([[sample_text]]).toDF("text")

result = pipeline.fit(df).transform(df)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_chemprot", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("ner")
    .setCaseSensitive(True)

val ner_converter = new NerConverter()
    .setInputCols(Array("document","token","ner"))
    .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(
                      document_assembler, 
                      tokenizer, 
                      tokenClassifier, 
                      ner_converter))

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
|Compatibility:|Healthcare NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|


## Data Source


This model is trained on a [ChemProt corpus](https://biocreative.bioinformatics.udel.edu/).


## Benchmarking


```bash
label       precision    recall  f1-score   support
B-CHEMICAL     0.93      0.79      0.85      8649
B-GENE-N       0.63      0.56      0.59      2752
B-GENE-Y       0.82      0.73      0.77      5490
I-CHEMICAL     0.90      0.79      0.84      1313
I-GENE-N       0.72      0.62      0.67      1993
I-GENE-Y       0.81      0.72      0.77      2420
accuracy       -         -         0.73     22617
macro-avg      0.75      0.74      0.75     22617
weighted-avg   0.83      0.73      0.78     22617
```
