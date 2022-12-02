---
layout: model
title: Detect Bacterial Species (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_bacteria
date: 2022-01-07
tags: [bacteria, bertfortokenclassification, ner, en, licensed]
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


Detect different types of species of bacteria in text using pretrained NER model. This model is trained with the `BertForTokenClassification` method from `transformers` library and imported into Spark NLP.


## Predicted Entities


`SPECIES`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bacteria_en_3.3.4_2.4_1641568604267.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols(["document"]) \
.setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassification.pretrained("bert_token_classifier_ner_bacteria", "en", "clinical/models")\
.setInputCols("token", "document")\
.setOutputCol("ner")\
.setCaseSensitive(True)

ner_converter = NerConverter()\
.setInputCols(["document","token","ner"])\
.setOutputCol("ner_chunk") 

pipeline = Pipeline(stages=[documentAssembler, 
tokenizer, 
tokenClassifier, 
ner_converter])

data = spark.createDataFrame([[ """Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents \
a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica \
sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T))."""
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

val tokenClassifier = MedicalBertForTokenClassification.pretrained("bert_token_classifier_ner_bacteria", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("ner")
.setCaseSensitive(True)

val ner_converter = new NerConverter()
.setInputCols(Array("document","token","ner"))
.setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents \
a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica \
sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).""").toDS.toDF("text")


val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_bacteria").predict("""Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents \
a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica \
sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).""")
```

</div>


## Results


```bash
+-------------------+----------------------+
|chunk              |ner_label             |
+-------------------+----------------------+
|erbA IRES          |Organism              |
|erbA/myb virus     |Organism              |
|erythroid cells    |Cell                  |
|bone marrow        |Multi-tissue_structure|
|blastoderm cultures|Cell                  |
|erbA/myb IRES virus|Organism              |
|erbA IRES virus    |Organism              |
|blastoderm         |Cell                  |
+-------------------+----------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bacteria|
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


Trained on Cancer Genetics (CG) task of the BioNLP Shared Task 2013. https://aclanthology.org/W13-2008/


## Benchmarking


```bash
label  precision    recall  f1-score   support
B-SPECIES       0.98      0.84      0.91       767
I-SPECIES       0.99      0.84      0.91      1043
accuracy         -         -       0.84      1810
macro-avg       0.85      0.89      0.87      1810
weighted-avg       0.99      0.84      0.91      1810
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjkyMzQ2NzE0XX0=
-->