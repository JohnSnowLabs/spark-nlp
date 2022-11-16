---
layout: model
title: Part of Speech for Amharic (pos_ud_att)
author: John Snow Labs
name: pos_ud_att
date: 2021-01-20
task: Part of Speech Tagging
language: am
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [am, pos, open_source]
supported: true
annotator: PerceptronModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates the part of speech of tokens in a text. The parts of speech annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 13 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

## Predicted Entities

| POS tag | Description                |
|---------|----------------------------|
| ADJ     |  adjective                 |
| ADP     |  adposition                |
| ADV     |  adverb                    |
| AUX     |  auxiliary                 |
| CCONJ   |  coordinating conjunction  |
| DET     |  determiner                |
| INTJ    |  interjection              |
| NOUN    |  noun                      |
| NUM     |  numeral                   |
| PART    |  particle                  |
| PRON    |  pronoun                   |
| PROPN   |  proper noun               |
| PUNCT   |  punctuation               |
| SCONJ   |  subordinating conjunction |
| SYM     |  symbol                    |
| VERB    |  verb                      |
| X       |  other                     |

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/GRAMMAR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_att_am_2.7.0_2.4_1611180723328.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
        
pos = PerceptronModel.pretrained("pos_ud_att", "am") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")

pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        posTagger
    ])

example = spark.createDataFrame([['ልጅ ኡ ን ሥራ ው ን አስጨርስ ኧው ኣል ኧሁ ።']], ["text"])

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
        
val pos = PerceptronModel.pretrained("pos_ud_att", "am")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))

val data = Seq("ልጅ ኡ ን ሥራ ው ን አስጨርስ ኧው ኣል ኧሁ ።").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["ልጅ ኡ ን ሥራ ው ን አስጨርስ ኧው ኣል ኧሁ ።"]
pos_df = nlu.load('am.pos').predict(text)
pos_df
```

</div>

## Results

```bash
+------------------------------+----------------------------------------------------------------+
|text                          |result                                                          |
+------------------------------+----------------------------------------------------------------+
|ልጅ ኡ ን ሥራ ው ን አስጨርስ ኧው ኣል ኧሁ ።|[NOUN, DET, PART, NOUN, DET, PART, VERB, PRON, AUX, PRON, PUNCT]|
+------------------------------+----------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_att|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[pos]|
|Language:|am|

## Data Source

The model was trained on the [Universal Dependencies](https://universaldependencies.org/) version 2.7.

Reference:

- Binyam Ephrem Seyoum ,Yusuke Miyao and Baye Yimam Mekonnen.2018.Universal Dependencies for Amharic. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), pp. 2216–2222, Miyazaki, Japan: European Language Resources Association (ELRA)

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|:------------:|:---------:|:------:|:--------:|:-------:|
|      ADJ     |    1.00   |  0.97  |   0.99   |   116   |
|      ADP     |    0.99   |  1.00  |   0.99   |   681   |
|      ADV     |    0.94   |  0.99  |   0.96   |    93   |
|      AUX     |    1.00   |  1.00  |   1.00   |   419   |
|     CCONJ    |    0.99   |  0.97  |   0.98   |    99   |
|      DET     |    0.99   |  1.00  |   0.99   |   485   |
|     INTJ     |    0.97   |  0.99  |   0.98   |    67   |
|     NOUN     |    0.99   |  1.00  |   1.00   |   1485  |
|      NUM     |    1.00   |  1.00  |   1.00   |    42   |
|     PART     |    1.00   |  1.00  |   1.00   |   875   |
|     PRON     |    1.00   |  1.00  |   1.00   |   2547  |
|     PROPN    |    1.00   |  0.99  |   0.99   |   236   |
|     PUNCT    |    1.00   |  1.00  |   1.00   |   1093  |
|     SCONJ    |    1.00   |  0.98  |   0.99   |   214   |
|     VERB     |    1.00   |  1.00  |   1.00   |   1552  |
|   accuracy   |           |        |   1.00   |  10004  |
|   macro avg  |    0.99   |  0.99  |   0.99   |  10004  |
| weighted avg |    1.00   |  1.00  |   1.00   |  10004  |
```