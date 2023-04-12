---
layout: model
title: Toxic content classifier for Russian
author: John Snow Labs
name: bert_sequence_classifier_toxicity
date: 2021-12-22
tags: [sentiment, bert, sequence, russian, ru, open_source]
task: Text Classification
language: ru
edition: Spark NLP 3.3.4
spark_version: 2.4
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been fine-tuned for the Russian language, leveraging `Bert` embeddings and `BertForSequenceClassification` for text classification purposes.

## Predicted Entities

`neutral`, `toxic`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_RU_TOXIC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_RU_TOXIC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_toxicity_ru_3.3.4_2.4_1640162987772.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_toxicity_ru_3.3.4_2.4_1640162987772.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler() \
.setInputCol('text') \
.setOutputCol('document')

tokenizer = Tokenizer() \
.setInputCols(['document']) \
.setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
.pretrained('bert_sequence_classifier_toxicity', 'ru') \
.setInputCols(['token', 'document']) \
.setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([["Ненавижу тебя, идиот."]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_toxicity", "ru")
.setInputCols("document", "token")
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["Ненавижу тебя, идиот."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("ru.classify.toxic").predict("""Ненавижу тебя, идиот.""")
```

</div>

## Results

```bash
['toxic']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_toxicity|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ru|
|Size:|665.1 MB|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/SkolkovoInstitute/russian_toxicity_classifier](https://huggingface.co/SkolkovoInstitute/russian_toxicity_classifier)

## Benchmarking

```bash
label         precision   recall   f1-score   support
neutral       0.98        0.99     0.98       21384
toxic         0.94        0.92     0.93       4886
accuracy      -           -        0.97       26270
macro-avg     0.96        0.96     0.96       26270
weighted-avg  0.97        0.97     0.97       26270
```
