---
layout: model
title: Detect Bacterial Species (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_bacteria
date: 2021-09-30
tags: [bacteria, bertfortokenclassification, ner, en, licensed]
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


Detect different types of species of bacteria in text using pretrained NER model. This model is trained with the `BertForTokenClassification` method from `transformers` library and imported into Spark NLP.


## Predicted Entities


`SPECIES`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bacteria_en_3.2.2_2.4_1632995062374.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bacteria_en_3.2.2_2.4_1632995062374.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

tokenClassifier = MedicalBertForTokenClassification.pretrained("bert_token_classifier_ner_bacteria", "en", "clinical/models")\
        .setInputCols("token", "document")\
        .setOutputCol("ner")\
        .setCaseSensitive(True)

ner_converter = NerConverter()\
        .setInputCols(["document","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

test_sentence = """Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents \
a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica \
sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T))."""

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [test_sentence]})))
```
```scala
val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val tokenizer = new Tokenizer()
        .setInputCols("document")
        .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_bacteria", "en", "clincal/models")
        .setInputCols(Array("token", "document"))
        .setOutputCol("ner")
        .setCaseSensitive(True)

val ner_converter = new NerConverter()
        .setInputCols(Array("document","token","ner"))
        .setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).""").toDS.toDF("text")

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
+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|SMSP (T)               |SPECIES  |
|Methanoregula formicica|SPECIES  |
|SMSP (T)               |SPECIES  |
+-----------------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bacteria|
|Compatibility:|Healthcare NLP 3.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|


## Data Source


Trained on a custom dataset by John Snow Labs.


## Benchmarking


```bash
label         precision    recall   f1-score   support
B-SPECIES       0.98        0.84      0.91       767
I-SPECIES       0.99        0.84      0.91      1043
accuracy         -           -        0.84      1810
macro-avg       0.85        0.89      0.87      1810
weighted-avg    0.99        0.84      0.91      1810
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NDk5NTU3NjBdfQ==
-->