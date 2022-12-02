---
layout: model
title: Part of Speech for Bhojpuri (pos_ud_bhtb)
author: John Snow Labs
name: pos_ud_bhtb
date: 2021-01-18
task: Part of Speech Tagging
language: bh
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [bho, bh, pos, open_source]
supported: true
annotator: PerceptronModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates the part of speech of tokens in a text. The parts of speech annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 14 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

## Predicted Entities

`ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `VERB`, and `X`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_bhtb_bh_2.7.0_2.4_1610989017843.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")
        
pos = PerceptronModel.pretrained("pos_ud_bhtb", "bh") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        pos
    ])

example = spark.createDataFrame([['ओहु लोग के मालूम बा कि श्लील होखते भोजपुरी के नींव हिल जाई ।']], ["text"])
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
        .setOutputCol("sentence")
        
val tokenizer = Tokenizer()
    .setInputCols(["sentence"])
    .setOutputCol("token")
        
val pos = PerceptronModel.pretrained("pos_ud_bhtb", "bh")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("ओहु लोग के मालूम बा कि श्लील होखते भोजपुरी के नींव हिल जाई ।").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["ओहु लोग के मालूम बा कि श्लील होखते भोजपुरी के नींव हिल जाई ।"]
pos_df = nlu.load('bh.pos').predict(text)
pos_df
```

</div>

## Results

```bash
+------------------------------------------------------------+----------------------------------------------------------------------------------+
|text                                                        |result                                                                            |
+------------------------------------------------------------+----------------------------------------------------------------------------------+
|ओहु लोग के मालूम बा कि श्लील होखते भोजपुरी के नींव हिल जाई ।|[DET, NOUN, ADP, NOUN, VERB, SCONJ, ADJ, VERB, PROPN, ADP, NOUN, VERB, AUX, PUNCT]|
+------------------------------------------------------------+----------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_bhtb|
|Compatibility:|Spark NLP 2.7.0+|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|bh|

## Data Source

The model was trained on the [Universal Dependencies](http://universaldependencies.org) version 2.7.


Reference:

  - Ojha, A. K., & Zeman, D. (2020). Universal Dependency Treebanks for Low-Resource Indian Languages: The Case of Bhojpuri. Proceedings of the WILDRE5{--} 5th Workshop on Indian Language Data: Resources and Evaluation.

## Benchmarking

```bash
|     pos     | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| ADJ          | 0.92      | 0.92   | 0.92     | 250     |
| ADP          | 0.95      | 0.96   | 0.96     | 989     |
| ADV          | 0.85      | 0.88   | 0.86     | 32      |
| AUX          | 0.93      | 0.95   | 0.94     | 355     |
| CCONJ        | 0.95      | 0.95   | 0.95     | 151     |
| DET          | 0.96      | 0.95   | 0.95     | 353     |
| INTJ         | 1.00      | 1.00   | 1.00     | 5       |
| NOUN         | 0.95      | 0.96   | 0.96     | 1854    |
| NUM          | 0.97      | 0.98   | 0.97     | 149     |
| PART         | 0.94      | 0.93   | 0.93     | 192     |
| PRON         | 0.95      | 0.94   | 0.95     | 335     |
| PROPN        | 0.94      | 0.94   | 0.94     | 419     |
| PUNCT        | 0.97      | 0.96   | 0.96     | 695     |
| SCONJ        | 1.00      | 0.96   | 0.98     | 118     |
| VERB         | 0.95      | 0.93   | 0.94     | 767     |
| X            | 0.50      | 1.00   | 0.67     | 1       |
| accuracy     |           |        | 0.95     | 6665    |
| macro avg    | 0.92      | 0.95   | 0.93     | 6665    |
| weighted avg | 0.95      | 0.95   | 0.95     | 6665    |
```
