---
layout: model
title: Named Entity Recognition for Japanese (FastText 300d)
author: John Snow Labs
name: ner_ud_gsd_cc_300d
date: 2021-09-09
tags: [ner, ja, open_source]
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

This model uses the pre-trained fasttext_ja_300d embeddings model from WordEmbeddings annotator as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`ORDINAL`, `PERSON`, `LAW`, `MOVEMENT`, `LOC`, `WORK_OF_ART`, `DATE`, `NORP`, `TITLE_AFFIX`, `QUANTITY`, `FAC`, `TIME`, `MONEY`, `LANGUAGE`, `GPE`, `EVENT`, `ORG`, `PERCENT`, `PRODUCT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ud_gsd_cc_300d_ja_3.2.2_3.0_1631189041655.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentence = SentenceDetector() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja") \
.setInputCols(["sentence"]) \
.setOutputCol("token")

embeddings = WordEmbeddings().pretrained("japanese_cc_300d", "ja") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("embeddings")

# Then the training can start
nerTagger = NerDLModel.pretrained("ner_ud_gsd_cc_300d", "ja") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")

pipeline = Pipeline().setStages([
documentAssembler,
sentence,
word_segmenter,
embeddings,
nerTagger
])

data = spark.createDataFrame([["宮本茂氏は、日本の任天堂のゲームプロデューサーです。"]]).toDF("text")
model = pipeline.fit(data)
result = model.transform(data)

result.selectExpr("explode(arrays_zip(token.result, ner.result))") \
.selectExpr("col['0'] as token", "col['1'] as ner") \
.show()
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, WordSegmenterModel}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
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

val embeddings = WordEmbeddingsModel.pretrained("japanese_cc_300d", "ja")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val nerTagger = NerDLModel.pretrained("ner_ud_gsd_cc_300d", "ja")
.setInputCols("sentence", "token", "embeddings")
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

result.selectExpr("explode(arrays_zip(token.result, ner.result))")
.selectExpr("col'0' as token", "col'1' as ner")
.show()
```


{:.nlu-block}
```python
import nlu
nlu.load("ja.ner.ud_gsd_cc_300d").predict("""explode(arrays_zip(token.result, ner.result))""")
```

</div>

## Results

```bash
+--------------+--------+
|         token|     ner|
+--------------+--------+
|          宮本|B-PERSON|
|            茂|I-PERSON|
|            氏|       O|
|            は|       O|
|            、|       O|
|          日本|   B-GPE|
|            の|       O|
|          任天|   B-FAC|
|            堂|   I-FAC|
|            の|       O|
|        ゲーム|       O|
|プロデューサー|       O|
|          です|       O|
|            。|       O|
+--------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ud_gsd_cc_300d|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ja|
|Dependencies:|japanese_cc_300d|

## Data Source

The model was trained on the Universal Dependencies, curated by Google. A NER version was created by megagonlabs:

https://github.com/megagonlabs/UD_Japanese-GSD

Reference:

Asahara, M., Kanayama, H., Tanaka, T., Miyao, Y., Uematsu, S., Mori, S., Matsumoto, Y., Omura, M., & Murawaki, Y. (2018). Universal Dependencies Version 2 for Japanese. In LREC-2018.

## Benchmarking

```bash
label  precision    recall  f1-score   support
DATE       0.91      0.92      0.92       206
EVENT       0.91      0.56      0.69        52
FAC       0.82      0.63      0.71        59
GPE       0.81      0.90      0.86       102
LANGUAGE       1.00      1.00      1.00         8
LAW       0.44      0.31      0.36        13
LOC       0.83      0.93      0.87        41
MONEY       0.80      1.00      0.89        20
MOVEMENT       0.38      0.55      0.44        11
NORP       0.98      0.81      0.88        57
O       0.99      0.99      0.99     11785
ORDINAL       0.79      0.94      0.86        32
ORG       0.82      0.74      0.78       179
PERCENT       1.00      1.00      1.00        16
PERSON       0.87      0.87      0.87       127
PRODUCT       0.61      0.72      0.66        50
QUANTITY       0.91      0.91      0.91       172
TIME       0.97      0.88      0.92        32
TITLE_AFFIX       0.81      0.92      0.86        24
WORK_OF_ART       0.71      0.83      0.77        48
accuracy          -         -      0.98     13034
macro-avg       0.82      0.82      0.81     13034
weighted-avg       0.98      0.98      0.98     13034
```