---
layout: model
title: BERT Sequence Classification - Detecting Hate Speech (bert_sequence_classifier_dehatebert_mono)
author: John Snow Labs
name: bert_sequence_classifier_dehatebert_mono
date: 2021-11-03
tags: [bert, hatespeech, en, english, sequence_classification, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is used detecting **hatespeech** in **English language**. The mono in the name refers to the monolingual setting, where the model is trained using only English language data. It is finetuned on multilingual bert model.
The model is trained with different learning rates and the best validation score achieved is 0.726030 for a learning rate of 2e-5. Training code can be found here https://github.com/punyajoy/DE-LIMIT

### For more details about our paper

Sai Saketh Aluru, Binny Mathew, Punyajoy Saha and Animesh Mukherjee. "[Deep Learning Models for Multilingual Hate Speech Detection](https://arxiv.org/abs/2004.06465)". Accepted at ECML-PKDD 2020.

~~~
@article{aluru2020deep,
title={Deep Learning Models for Multilingual Hate Speech Detection},
author={Aluru, Sai Saket and Mathew, Binny and Saha, Punyajoy and Mukherjee, Animesh},
journal={arXiv preprint arXiv:2004.06465},
year={2020}
}
~~~

## Predicted Entities

`NON_HATE`, `HATE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_dehatebert_mono_en_3.3.2_3.0_1635937844054.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('bert_sequence_classifier_dehatebert_mono', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler,
tokenizer,
sequenceClassifier
])

example = spark.createDataFrame([['I love you!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_dehatebert_mono", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I love you!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.bert_sequence.dehatebert_mono").predict("""I love you!""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_dehatebert_mono|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|en|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english)