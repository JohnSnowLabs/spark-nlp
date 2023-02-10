---
layout: model
title: Named Entity Recognition for Japanese (BERT Base Japanese)
author: John Snow Labs
name: ner_ud_gsd_bert_base_japanese
date: 2021-09-16
tags: [ja, ner, open_sourve, open_source]
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

This model uses the pretrained BertEmbeddings embeddings "bert_base_ja" as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`ORDINAL`, `PERSON`, `LAW`, `MOVEMENT`, `LOC`, `WORK_OF_ART`, `DATE`, `NORP`, `TITLE_AFFIX`, `QUANTITY`, `FAC`, `TIME`, `MONEY`, `LANGUAGE`, `GPE`, `EVENT`, `ORG`, `PERCENT`, `PRODUCT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ud_gsd_bert_base_japanese_ja_3.2.2_3.0_1631804789491.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_ud_gsd_bert_base_japanese_ja_3.2.2_3.0_1631804789491.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = BertEmbeddings.pretrained("bert_base_japanese", "ja") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")

nerTagger = NerDLModel.pretrained("ner_ud_gsd_bert_base_japanese", "ja") \
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
import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
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

val embeddings = BertEmbeddings.pretrained("bert_base_japanese", "ja")
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")

val nerTagger = NerDLModel.pretrained("ner_ud_gsd_bert_base_japanese", "ja")
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

result.selectExpr("explode(arrays_zip(token.result, ner.result))").show()
```
</div>

## Results

```bash
# +-------------------+
# |                col|
# +-------------------+
# |   {宮本, B-PERSON}|
# |     {茂, I-PERSON}|
# |            {氏, O}|
# |            {は, O}|
# |            {、, O}|
# |      {日本, B-GPE}|
# |            {の, O}|
# |      {任天, B-ORG}|
# |        {堂, I-ORG}|
# |            {の, O}|
# |        {ゲーム, O}|
# |{プロデューサー, O}|
# |          {です, O}|
# |            {。, O}|
# +-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ud_gsd_bert_base_japanese|
|Type:|ner|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ja|
|Dependencies:|bert_base_ja|

## Data Source

The model was trained on the Universal Dependencies, curated by Google. A NER version was created by megagonlabs:

https://github.com/megagonlabs/UD_Japanese-GSD

Reference:

    Asahara, M., Kanayama, H., Tanaka, T., Miyao, Y., Uematsu, S., Mori, S., Matsumoto, Y., Omura, M., & Murawaki, Y. (2018). Universal Dependencies Version 2 for Japanese. In LREC-2018.

## Benchmarking

```bash
       label  precision    recall  f1-score   support
    CARDINAL       0.00      0.00      0.00         0
        DATE       0.95      0.96      0.96       206
       EVENT       0.84      0.50      0.63        52
         FAC       0.75      0.71      0.73        59
         GPE       0.79      0.76      0.78       102
    LANGUAGE       1.00      1.00      1.00         8
         LAW       1.00      0.31      0.47        13
         LOC       0.89      0.83      0.86        41
       MONEY       1.00      1.00      1.00        20
    MOVEMENT       1.00      0.18      0.31        11
        NORP       0.85      0.82      0.84        57
           O       0.99      0.99      0.99     11785
     ORDINAL       0.81      0.94      0.87        32
         ORG       0.78      0.65      0.71       179
     PERCENT       0.89      1.00      0.94        16
      PERSON       0.76      0.84      0.80       127
     PRODUCT       0.62      0.68      0.65        50
    QUANTITY       0.92      0.94      0.93       172
        TIME       0.97      0.88      0.92        32
 TITLE_AFFIX       0.89      0.71      0.79        24
 WORK_OF_ART       0.66      0.73      0.69        48
    accuracy          -         -      0.97     13034
   macro-avg       0.83      0.73      0.75     13034
weighted-avg       0.97      0.97      0.97     13034
```