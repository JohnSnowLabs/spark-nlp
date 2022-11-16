---
layout: model
title: BERT Sequence Classification - Detecting Hate Speech (bert_sequence_classifier_hatexplain)
author: John Snow Labs
name: bert_sequence_classifier_hatexplain
date: 2021-11-06
tags: [bert_for_sequence_classification, hate, hate_speech, speech, offensive, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is imported from `Hugging Face-models`and it is used for classifying a text as `Hate speech`, `Offensive`, or `Normal`. The model is trained using data from Gab and Twitter and Human Rationales were included as part of the training data to boost the performance.

- Citing :
```bash
@article{mathew2020hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  journal={arXiv preprint arXiv:2012.10289},
  year={2020}
}
```

## Predicted Entities

`hate speech`, `normal`, `offensive`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_hatexplain_en_3.3.2_2.4_1636214446271.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained('bert_sequence_classifier_hatexplain', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['I love you very much!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_hatexplain", "en")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["I love you very much!"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
['normal']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_hatexplain|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[label]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain](https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain)

## Benchmarking

```bash

+-------+------------+--------+
| Acc   | Macro F1   | AUROC  |
+-------+------------+--------+
| 0.698 | 0.687      | 0.851  |
+-------+------------+--------+

```
