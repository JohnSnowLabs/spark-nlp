---
layout: model
title: Part of Speech for Japanese
author: John Snow Labs
name: pos_ud_gsd
date: 2021-01-03
task: Part of Speech Tagging
language: ja
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [pos, ja, open_source]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates the part of speech of tokens in a text. The parts of speech annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 13 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_ja_2.7.0_2.4_1609700150824.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence_detector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")\
.setInputCols(["sentence"])\
.setOutputCol("token")

pos = PerceptronModel.pretrained("pos_ud_gsd", "ja") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
word_segmenter,
posTagger
])

example = spark.createDataFrame([['院長と話をしたところ、腰痛治療も得意なようです。']], ["text"])

result = pipeline.fit(example).transform(example)

```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = SentenceDetector()
.setInputCols(["document"])
.setOutputCol("sentence")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")
.setInputCols(["sentence"])
.setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_ud_gsd", "ja")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, pos))

val data = Seq("院長と話をしたところ、腰痛治療も得意なようです。").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""5月13日に放送されるフジテレビ系「僕らの音楽」にて、福原美穂とAIという豪華共演が決定した。"""]
pos_df = nlu.load('ja.pos.ud_gsd').predict(text, output_level='token')
pos_df
```

</div>

## Results

```bash
+------+-----+
|token |pos  |
+------+-----+
|院長  |NOUN |
|と    |ADP  |
|話    |NOUN |
|を    |ADP  |
|し    |VERB |
|た    |AUX  |
|ところ|NOUN |
|、    |PUNCT|
|腰痛  |NOUN |
|治療  |NOUN |
|も    |ADP  |
|得意  |ADJ  |
|な    |AUX  |
|よう  |AUX  |
|です  |AUX  |
|。    |PUNCT|
+------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_gsd|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|ja|

## Data Source

The model was trained on the [Universal Dependencies](https://universaldependencies.org/), curated by Google.

Reference:

> Asahara, M., Kanayama, H., Tanaka, T., Miyao, Y., Uematsu, S., Mori, S., Matsumoto, Y., Omura, M., & Murawaki, Y. (2018). 
Universal Dependencies Version 2 for Japanese. In LREC-2018.

## Benchmarking

```bash
| pos_tag      | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.90      | 0.78   | 0.84     | 350     |
| ADP          | 0.98      | 0.99   | 0.99     | 2804    |
| ADV          | 0.87      | 0.65   | 0.74     | 220     |
| AUX          | 0.95      | 0.98   | 0.96     | 1768    |
| CCONJ        | 0.97      | 0.93   | 0.95     | 42      |
| DET          | 1.00      | 1.00   | 1.00     | 66      |
| INTJ         | 0.00      | 0.00   | 0.00     | 1       |
| NOUN         | 0.93      | 0.98   | 0.95     | 3692    |
| NUM          | 0.99      | 0.98   | 0.99     | 251     |
| PART         | 0.96      | 0.83   | 0.89     | 128     |
| PRON         | 0.97      | 0.94   | 0.95     | 101     |
| PROPN        | 0.92      | 0.70   | 0.79     | 313     |
| PUNCT        | 1.00      | 1.00   | 1.00     | 1294    |
| SCONJ        | 0.97      | 0.94   | 0.96     | 682     |
| SYM          | 0.99      | 1.00   | 0.99     | 67      |
| VERB         | 0.96      | 0.92   | 0.94     | 1255    |
| accuracy     | 0.96      | 13034  |          |         |
| macro avg    | 0.90      | 0.85   | 0.87     | 13034   |
| weighted avg | 0.96      | 0.96   | 0.95     | 13034   |
```
