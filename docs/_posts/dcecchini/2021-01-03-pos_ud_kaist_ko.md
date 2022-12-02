---
layout: model
title: Part of Speech for Korean
author: John Snow Labs
name: pos_ud_kaist
date: 2021-01-03
task: Part of Speech Tagging
language: ko
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [pos, ko, open_source]
supported: true
annotator: PerceptronModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates the part of speech of tokens in a text. The parts of speech annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 13 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

## Predicted Entities

`ADJ`, `ADP`, `ADV`, `AUX`, `CONJ`, `DET`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SYM`, `VERB`, and `X`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_kaist_ko_2.7.0_2.4_1609701893746.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline after tokenization.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence_detector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_kaist_ud", "ko")\
.setInputCols(["sentence"])\
.setOutputCol("token")

pos = PerceptronModel.pretrained("pos_ud_kaist", "ko") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
word_segmenter,
posTagger
])

example = spark.createDataFrame([['비파를탄주하는그늙은명인의시는아름다운화음이었고완벽한음악으로순간적인조화를이룬세계의울림이었다.']], ["text"])

result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_kaist_ud", "ko")
.setInputCols(["sentence"])
.setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_ud_kaist", "ko")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, pos))

val data = Seq("비파를탄주하는그늙은명인의시는아름다운화음이었고완벽한음악으로순간적인조화를이룬세계의울림이었다.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""비파를탄주하는그늙은명인의시는아름다운화음이었고완벽한음악으로순간적인조화를이룬세계의울림이었다."""]
pos_df = nlu.load('ko.pos.ud_kaist').predict(text, output_level='token')
pos_df
```

</div>

## Results

```bash
+----------+-----+
|token     |pos  |
+----------+-----+
|비파를    |NOUN |
|탄주하는  |VERB |
|그        |DET  |
|늙은      |VERB |
|명인의    |NOUN |
|시는      |NOUN |
|아름다운  |ADJ  |
|화음이었고|CCONJ|
|완벽한    |VERB |
|음악으로  |NOUN |
|순간적인  |VERB |
|조화를    |NOUN |
|이룬      |VERB |
|세계의    |NOUN |
|울림이었다|VERB |
|.         |PUNCT|
+----------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_kaist|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|ko|

## Data Source

The model was trained in the Universal Dependencies, curated by the Korea Advanced Institute of Science and Technology (KAIST)

Reference:

> Building Universal Dependency Treebanks in Korean, Jayeol Chun, Na-Rae Han, Jena D. Hwang, and Jinho D. Choi. 
In Proceedings of the 11th International Conference on Language Resources and Evaluation, LREC'18, Miyazaki, Japan, 2018.

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.90      | 0.81   | 0.85     | 1180    |
| ADP          | 0.95      | 0.97   | 0.96     | 160     |
| ADV          | 0.89      | 0.82   | 0.85     | 4156    |
| AUX          | 0.85      | 0.84   | 0.84     | 1074    |
| CCONJ        | 0.82      | 0.71   | 0.76     | 1471    |
| DET          | 0.92      | 0.91   | 0.91     | 272     |
| INTJ         | 0.00      | 0.00   | 0.00     | 3       |
| NOUN         | 0.80      | 0.93   | 0.86     | 8338    |
| NUM          | 0.86      | 0.91   | 0.88     | 631     |
| PART         | 0.00      | 0.00   | 0.00     | 18      |
| PRON         | 0.96      | 0.91   | 0.93     | 405     |
| PROPN        | 0.85      | 0.58   | 0.69     | 1377    |
| PUNCT        | 1.00      | 1.00   | 1.00     | 3109    |
| SCONJ        | 0.84      | 0.72   | 0.78     | 1547    |
| SYM          | 1.00      | 0.98   | 0.99     | 115     |
| VERB         | 0.87      | 0.87   | 0.87     | 4378    |
| X            | 0.94      | 0.71   | 0.81     | 132     |
| accuracy     | 0.86      | 28366  |          |         |
| macro avg    | 0.79      | 0.74   | 0.76     | 28366   |
| weighted avg | 0.86      | 0.86   | 0.86     | 28366   |
```
