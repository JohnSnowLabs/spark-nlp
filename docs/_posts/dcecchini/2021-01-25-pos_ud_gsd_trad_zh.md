---
layout: model
title: Part of Speech for Traditional Chinese
author: John Snow Labs
name: pos_ud_gsd_trad
date: 2021-01-25
task: Part of Speech Tagging
language: zh
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [pos, zh, open_source]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_trad_zh_2.7.0_2.4_1611578220288.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_trad_zh_2.7.0_2.4_1611578220288.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud_trad", "zh")\
        .setInputCols(["sentence"])\
        .setOutputCol("token")
        
pos = PerceptronModel.pretrained("pos_ud_gsd_trad", "zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        word_segmenter,
        posTagger
    ])

example = spark.createDataFrame([['然而，這樣的處理也衍生了一些問題。']], ["text"])

result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
        .setOutputCol("sentence")
        
val word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud_trad", "zh")
        .setInputCols(["sentence"])
        .setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_ud_gsd_trad", "zh")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, pos))

val data = Seq("然而，這樣的處理也衍生了一些問題。").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""然而，這樣的處理也衍生了一些問題。"""]
pos_df = nlu.load('zh.pos.ud_gsd_trad').predict(text, output_level = "token")
pos_df
```

</div>

## Results

```bash
+------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|text                                                                          |result                                                                                                           |
+------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
|然而 ， 這樣 的 處理 也 衍生 了 一些 問題 。                                  |[ADV, PUNCT, PRON, PART, NOUN, ADV, VERB, PART, ADJ, NOUN, PUNCT]                                                |
+------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_gsd_trad|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|zh|

## Data Source

The model was trained on the [Universal Dependencies](https://universaldependencies.org/) for Traditional Chinese annotated and converted by Google.

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.70      | 0.68   | 0.69     | 272     |
| ADP          | 0.85      | 0.86   | 0.85     | 535     |
| ADV          | 0.90      | 0.90   | 0.90     | 549     |
| AUX          | 0.88      | 0.88   | 0.88     | 281     |
| CCONJ        | 0.92      | 0.87   | 0.89     | 191     |
| DET          | 0.93      | 0.93   | 0.93     | 138     |
| NOUN         | 0.88      | 0.92   | 0.90     | 3312    |
| NUM          | 0.98      | 0.99   | 0.98     | 653     |
| PART         | 0.97      | 0.94   | 0.95     | 1359    |
| PRON         | 0.97      | 0.97   | 0.97     | 168     |
| PROPN        | 0.89      | 0.84   | 0.86     | 1006    |
| PUNCT        | 1.00      | 1.00   | 1.00     | 1688    |
| SYM          | 1.00      | 1.00   | 1.00     | 3       |
| VERB         | 0.86      | 0.83   | 0.85     | 1769    |
| X            | 1.00      | 0.88   | 0.93     | 88      |
| accuracy     |           |        | 0.91     | 12012   |
| macro avg    | 0.91      | 0.90   | 0.91     | 12012   |
| weighted avg | 0.91      | 0.91   | 0.91     | 12012   |
```
