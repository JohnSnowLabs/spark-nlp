---
layout: model
title: Named Entity Recognition for Japanese (XLM-RoBERTa)
author: John Snow Labs
name: ner_ud_gsd_xlm_roberta_base
date: 2021-09-15
tags: [ja, ner, open_source]
task: Named Entity Recognition
language: ja
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.

This model uses the pretrained XlmRoBertaEmbeddings embeddings "xlm_roberta_base" as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`ORDINAL`, `PERSON`, `LAW`, `MOVEMENT`, `LOC`, `WORK_OF_ART`, `DATE`, `NORP`, `TITLE_AFFIX`, `QUANTITY`, `FAC`, `TIME`, `MONEY`, `LANGUAGE`, `GPE`, `EVENT`, `ORG`, `PERCENT`, `PRODUCT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ud_gsd_xlm_roberta_base_ja_3.2.2_3.0_1631696644878.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *
from sparknlp.training import *

documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja") \
.setInputCols(["sentence"]) \
.setOutputCol("token")

embeddings = XlmRoBertaEmbeddings.pretrained() \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

nerTagger = NerDLModel.pretrained("ner_ud_gsd_xlm_roberta_base", "ja") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")

pipeline = Pipeline().setStages(
[
documentAssembler,
sentence,
word_segmenter,
embeddings,
nerTagger,
]
)

data = spark.createDataFrame([["宮本茂氏は、日本の任天堂のゲームプロデューサーです。"]]).toDF("text")
model = pipeline.fit(data)
result = model.transform(data)
result.selectExpr("explode(arrays_zip(token.result, ner.result))").show()
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, WordSegmenterModel}
import com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence = new SentenceDetector()
.setInputCols("document")
.setOutputCol("sentence")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")
.setInputCols("sentence")
.setOutputCol("token")

val embeddings = XlmRoBertaEmbeddings.pretrained("japanese_cc_300d", "ja")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val nerTagger = NerDLModel.pretrained("ner_ud_gsd_xlm_roberta_base", "ja")
.setInputCols("sentence", "token")
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
sentence,
word_segmenter,
embeddings,
nerTagger
))

val data = Seq("宮本茂氏は、日本の任天堂のゲームプロデューサーです。").toDF("text")
val model = pipeline.fit(data)
val result = model.transform(data)

result.selectExpr("explode(arrays_zip(token.result, ner.result))").show()
```


{:.nlu-block}
```python
import nlu
nlu.load("ja.ner.ud_gsd_xlm_roberta_base").predict("""explode(arrays_zip(token.result, ner.result))""")
```

</div>

## Results

```bash
+-------------------+                                                           
|                col|
+-------------------+
|   {宮本, B-PERSON}|
|     {茂, I-PERSON}|
|            {氏, O}|
|            {は, O}|
|            {、, O}|
|      {日本, B-GPE}|
|            {の, O}|
|      {任天, B-ORG}|
|        {堂, I-ORG}|
|            {の, O}|
|        {ゲーム, O}|
|{プロデューサー, O}|
|          {です, O}|
|            {。, O}|
+-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ud_gsd_xlm_roberta_base|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ja|
|Dependencies:|xlm_roberta_base|

## Data Source

The model was trained on the Universal Dependencies, curated by Google. A NER version was created by megagonlabs:

https://github.com/megagonlabs/UD_Japanese-GSD

Reference:

Asahara, M., Kanayama, H., Tanaka, T., Miyao, Y., Uematsu, S., Mori, S., Matsumoto, Y., Omura, M., & Murawaki, Y. (2018). Universal Dependencies Version 2 for Japanese. In LREC-2018.

## Benchmarking

```bash
label  precision    recall  f1-score   support
DATE       0.93      0.97      0.95       206
EVENT       0.78      0.48      0.60        52
FAC       0.80      0.68      0.73        59
GPE       0.88      0.81      0.85       102
LANGUAGE       1.00      1.00      1.00         8
LAW       0.82      0.69      0.75        13
LOC       0.87      0.83      0.85        41
MONEY       1.00      1.00      1.00        20
MOVEMENT       0.67      0.55      0.60        11
NORP       0.84      0.86      0.85        57
O       0.99      0.99      0.99     11785
ORDINAL       0.94      0.94      0.94        32
ORG       0.71      0.78      0.74       179
PERCENT       1.00      1.00      1.00        16
PERSON       0.89      0.90      0.89       127
PRODUCT       0.56      0.68      0.61        50
QUANTITY       0.92      0.96      0.94       172
TIME       0.91      1.00      0.96        32
TITLE_AFFIX       0.86      0.75      0.80        24
WORK_OF_ART       0.87      0.85      0.86        48
accuracy          -         -      0.98     13034
macro-avg       0.86      0.84      0.85     13034
weighted-avg       0.98      0.98      0.98     13034
```