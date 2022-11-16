---
layout: model
title: Part of Speech for Bengali (pos_msri)
author: John Snow Labs
name: pos_msri
date: 2021-01-20
task: Part of Speech Tagging
language: bn
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [bn, pos, open_source]
supported: true
annotator: PerceptronModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates the part of speech of tokens in a text. The parts of speech annotated include NN (noun), CC (Conjuncts  - coordinating and subordinating), and 26 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

## Predicted Entities

`BM` (Not Documented), `CC (Conjuncts, Coordinating and Subordinating)`, `CL (Clitics)`, `DEM (Demonstratives)`, `INJ (Interjection)`, `INTF (Intensifier)`, `JJ (Adjective)`, `NEG (Negative)`, `NN (Noun)`, `NNC (Compound Nouns)`, `NNP (Proper Noun)`, `NST (Preposition of Direction)`, `PPR (Postposition)`, `PRP (Pronoun)`, `PSP (Preprosition)`, `QC (Cardinal Number)`, `QF (Quantifiers)`, `QO (Ordinal Numbers)`, `RB (Adverb)`, `RDP (Not Documented)`, `RP (Particle)`, `SYM (Special Symbol)`, `UT (Not Documented)`, `VAUX (Verb Auxiliary)`, `VM (Verb)`, `WQ (wh- qualifier)`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_msri_bn_2.7.0_2.4_1611173659719.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

pos = PerceptronModel.pretrained("pos_msri", "bn") \
.setInputCols(["document", "token"]) \
.setOutputCol("pos")

pipeline = Pipeline(stages=[
document_assembler,
sentence_detector,
tokenizer,
posTagger
])

example = spark.createDataFrame([["বাসস্থান-ঘরগৃহস্থালি তোড়া ভাষায় গ্রামকেও বলে  মোদ ' ৷"]], ["text"])

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

val pos = PerceptronModel.pretrained("pos_lst20", "th")
.setInputCols(Array("document", "token"))
.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))

val data = Seq("বাসস্থান-ঘরগৃহস্থালি তোড়া ভাষায় গ্রামকেও বলে ` মোদ ' ৷").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["বাসস্থান-ঘরগৃহস্থালি তোড়া ভাষায় গ্রামকেও বলে ` মোদ ' ৷"]
pos_df = nlu.load('bn.pos').predict(text, output_level = "token")
pos_df
```

</div>

## Results

```bash
+------------------------------------------------------+----------------------------------------+
|text                                                  |result                                  |
+------------------------------------------------------+----------------------------------------+
|বাসস্থান-ঘরগৃহস্থালি তোড়া ভাষায় গ্রামকেও বলে ` মোদ ' ৷|[NN, NNP, NN, NN, VM, SYM, NN, SYM, SYM]|
+------------------------------------------------------+----------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_msri|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|bn|

## Data Source

The model was trained on the _Indian Language POS-Tagged Corpus_ from [NLTK](http://www.nltk.org) collected by A Kumaran (Microsoft Research, India).

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| BM           | 1.00      | 1.00   | 1.00     | 1       |
| CC           | 0.99      | 0.99   | 0.99     | 390     |
| CL           | 1.00      | 1.00   | 1.00     | 2       |
| DEM          | 0.98      | 0.99   | 0.98     | 139     |
| INJ          | 0.92      | 0.85   | 0.88     | 13      |
| INTF         | 1.00      | 1.00   | 1.00     | 55      |
| JJ           | 0.99      | 0.99   | 0.99     | 688     |
| NEG          | 0.99      | 0.98   | 0.99     | 135     |
| NN           | 0.99      | 0.99   | 0.99     | 2996    |
| NNC          | 1.00      | 1.00   | 1.00     | 4       |
| NNP          | 0.97      | 0.98   | 0.97     | 528     |
| NST          | 1.00      | 1.00   | 1.00     | 156     |
| PPR          | 1.00      | 1.00   | 1.00     | 1       |
| PRP          | 0.98      | 0.98   | 0.98     | 685     |
| PSP          | 0.99      | 0.99   | 0.99     | 250     |
| QC           | 0.99      | 0.99   | 0.99     | 193     |
| QF           | 0.98      | 0.98   | 0.98     | 187     |
| QO           | 1.00      | 1.00   | 1.00     | 22      |
| RB           | 0.99      | 0.99   | 0.99     | 187     |
| RDP          | 1.00      | 0.98   | 0.99     | 44      |
| RP           | 0.99      | 0.96   | 0.97     | 79      |
| SYM          | 0.97      | 0.98   | 0.98     | 1413    |
| UNK          | 1.00      | 1.00   | 1.00     | 1       |
| UT           | 1.00      | 1.00   | 1.00     | 18      |
| VAUX         | 0.97      | 0.97   | 0.97     | 400     |
| VM           | 0.99      | 0.98   | 0.98     | 1393    |
| WQ           | 1.00      | 0.99   | 0.99     | 71      |
| XC           | 0.98      | 0.97   | 0.97     | 219     |
| accuracy     |           |        | 0.98     | 10270   |
| macro avg    | 0.99      | 0.98   | 0.99     | 10270   |
| weighted avg | 0.98      | 0.98   | 0.98     | 10270   |
```