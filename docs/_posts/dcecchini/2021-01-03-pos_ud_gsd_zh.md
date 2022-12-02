---
layout: model
title: Part of Speech for Chinese
author: John Snow Labs
name: pos_ud_gsd
date: 2021-01-03
task: Part of Speech Tagging
language: zh
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [pos, zh, cn, open_source]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_zh_2.7.0_2.4_1609699328856.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")\
        .setInputCols(["sentence"])\
        .setOutputCol("token")
        
pos = PerceptronModel.pretrained("pos_ud_gsd", "zh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        word_segmenter,
        posTagger
    ])

example = spark.createDataFrame([['然而，这样的处理也衍生了一些问题。']], ["text"])

result = pipeline.fit(example).transform(example)

```
```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
        .setOutputCol("sentence")
        
val word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")
        .setInputCols(["sentence"])
        .setOutputCol("token")

val pos = PerceptronModel.pretrained("pos_ud_gsd", "zh")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, pos))

val data = Seq("然而，这样的处理也衍生了一些问题。").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""然而，这样的处理也衍生了一些问题。"""]
pos_df = nlu.load('zh.pos.ud_gsd').predict(text, output_level='token')
pos_df
```

</div>

## Results

```bash
+-----+-----+
|token|pos  |
+-----+-----+
|然而 |ADV  |
|,    |PUNCT|
|这样 |PRON |
|的   |PART |
|处理 |NOUN |
|也   |ADV  |
|衍生 |VERB |
|了   |PART |
|一些 |ADJ  |
|问题 |NOUN |
|。   |PUNCT|
+-----+-----+
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
|Language:|zh|

## Data Source

The model was trained on the [Universal Dependencies (UD)](https://universaldependencies.org/) for Chinese (GNU license) curated by Google (Simplified Chinese)

Reference:

     > Zeman, Daniel; Nivre, Joakim; Abrams, Mitchell; et al., 2020, Universal Dependencies 2.7, LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University, http://hdl.handle.net/11234/1-3424.

## Benchmarking

```bash
| pos_tag          | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.68      | 0.67   | 0.67     | 271     |
| ADP          | 0.85      | 0.85   | 0.85     | 513     |
| ADV          | 0.90      | 0.91   | 0.91     | 549     |
| AUX          | 0.97      | 0.96   | 0.96     | 91      |
| CONJ         | 0.93      | 0.86   | 0.90     | 191     |
| DET          | 0.93      | 0.93   | 0.93     | 138     |
| NOUN         | 0.89      | 0.92   | 0.90     | 3310    |
| NUM          | 0.98      | 0.98   | 0.98     | 653     |
| PART         | 0.95      | 0.94   | 0.94     | 1346    |
| PRON         | 0.99      | 0.97   | 0.98     | 168     |
| PROPN        | 0.88      | 0.85   | 0.86     | 1006    |
| PUNCT        | 1.00      | 1.00   | 1.00     | 1688    |
| SYM          | 1.00      | 1.00   | 1.00     | 3       |
| VERB         | 0.88      | 0.86   | 0.87     | 1981    |
| X            | 0.99      | 0.79   | 0.88     | 104     |
| accuracy     | 0.91      | 12012  |          |         |
| macro avg    | 0.92      | 0.90   | 0.91     | 12012   |
| weighted avg | 0.91      | 0.91   | 0.91     | 12012   |
```
