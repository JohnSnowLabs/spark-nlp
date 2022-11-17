---
layout: model
title: Bert for Sequence Classification (Clinical Question vs Statement)
author: John Snow Labs
name: bert_sequence_classifier_question_statement_clinical
date: 2021-11-05
tags: [question, statement, clinical, en, licensed]
task: Text Classification
language: en
edition: Healthcare NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Trained to add sentence classifying capabilities to distinguish between Question vs Statements in clinical domain.


This model was imported from Hugging Face (https://huggingface.co/shahrukhx01/question-vs-statement-classifier), trained based on Haystack (https://github.com/deepset-ai/haystack/issues/611) and finetuned by John Snow Labs with in-house clinical annotations.


## Predicted Entities


`question`, `statement`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_question_statement_clinical_en_3.3.2_3.0_1636106577489.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
.setInputCols("sentence")\
.setOutputCol("token")

seq = nlp.BertForSequenceClassification.pretrained('bert_sequence_classifier_question_statement_clinical', 'en', 'clinical/models')\
.setInputCols(["token", "sentence"])\
.setOutputCol("label")\
.setCaseSensitive(True)

pipeline = Pipeline(stages = [
documentAssembler,
sentenceDetector,
tokenizer,
seq])

test_sentences = [["""Hello I am going to be having a baby throughand have just received my medical results before I have my tubes tested. I had the tests on day 23 of my cycle. My progresterone level is 10. What does this mean? What does progesterone level of 10 indicate?
Your progesterone report is perfectly normal. We expect this result on day 23rd of the cycle.So there's nothing to worry as it's perfectly alright"""]]

data = spark.createDataFrame(test_sentences).toDF("text")

res = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols("sentence")
.setOutputCol("token")

val seq = BertForSequenceClassification.pretrained("bert_sequence_classifier_question_statement_clinical", "en", "clinical/models")
.setInputCols(Array("token", "sentence"))
.setOutputCol("label")
.setCaseSensitive(True)

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
sentenceDetector,
tokenizer,
seq))

val test_sentences = """Hello I am going to be having a baby throughand have just received my medical results before I have my tubes tested. I had the tests on day 23 of my cycle. My progresterone level is 10. What does this mean? What does progesterone level of 10 indicate? Your progesterone report is perfectly normal. We expect this result on day 23rd of the cycle.So there's nothing to worry as it's perfectly alright"""

val example = Seq(test_sentences).toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.bert_sequence.question_statement_clinical").predict("""Hello I am going to be having a baby throughand have just received my medical results before I have my tubes tested. I had the tests on day 23 of my cycle. My progresterone level is 10. What does this mean? What does progesterone level of 10 indicate?
Your progesterone report is perfectly normal. We expect this result on day 23rd of the cycle.So there's nothing to worry as it's perfectly alright""")
```

</div>


## Results


```bash
+--------------------------------------------------------------------------------------------------------------------+---------+
|sentence                                                                                                            |label    |
+--------------------------------------------------------------------------------------------------------------------+---------+
|Hello I am going to be having a baby throughand have just received my medical results before I have my tubes tested.|statement|
|I had the tests on day 23 of my cycle.                                                                              |statement|
|My progresterone level is 10.                                                                                       |statement|
|What does this mean?                                                                                                |question |
|What does progesterone level of 10 indicate?                                                                        |question |
|Your progesterone report is perfectly normal. We expect this result on day 23rd of the cycle.                       |statement|
|So there's nothing to worry as it's perfectly alright                                                               |statement|
+--------------------------------------------------------------------------------------------------------------------+---------
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_question_statement_clinical|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[label]|
|Language:|en|
|Case sensitive:|true|


## Data Source


For generic domain training:
https://github.com/deepset-ai/haystack/issues/611


For finetuning in clinical domain, in house JSL annotations based on clinical Q&A.


## Benchmarking


```bash
label  precision    recall  f1-score   support
question       0.97      0.94      0.96       243
statement       0.98      0.99      0.99       729
accuracy       -         -         0.98       972
macro-avg       0.98      0.97      0.97       972
weighted-avg       0.98      0.98      0.98       972
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjA0Mjk4ODkyXX0=
-->