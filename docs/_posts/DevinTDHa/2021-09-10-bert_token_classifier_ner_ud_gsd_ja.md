---
layout: model
title: Named Entity Recognition for Japanese (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_ner_ud_gsd
date: 2021-09-10
tags: [ner, ja, open_source, bert]
task: Named Entity Recognition
language: ja
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations. It has BERT embeddings integrated with an attached dense layer at the end, to classify tokens directly. Word segmentation is needed to extract the tokens.

The model uses BERT embeddings from https://github.com/cl-tohoku/bert-japanese.

## Predicted Entities

`ORDINAL`, `PERSON`, `LAW`, `MOVEMENT`, `LOC`, `WORK_OF_ART`, `DATE`, `NORP`, `TITLE_AFFIX`, `QUANTITY`, `FAC`, `TIME`, `MONEY`, `LANGUAGE`, `GPE`, `EVENT`, `ORG`, `PERCENT`, `PRODUCT`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_ner_ud_gsd_ja_3.2.2_3.0_1631279615344.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_ner_ud_gsd_ja_3.2.2_3.0_1631279615344.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentenceDetector = SentenceDetector() \
.setInputCols("document") \
.setOutputCol("sentence")

word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja") \
.setInputCols(["sentence"]) \
.setOutputCol("token")

nerTagger = BertForTokenClassification \
.pretrained("bert_token_classifier_ner_ud_gsd", "ja") \
.setInputCols(["sentence",'token']) \
.setOutputCol("ner")

pipeline = Pipeline().setStages([
documentAssembler,
sentenceDetector,
word_segmenter,
nerTagger
])

data = spark.createDataFrame([["宮本茂氏は、日本の任天堂のゲームプロデューサーです。"]]).toDF("text")
model = pipeline.fit(data)
result = model.transform(data)
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{SentenceDetector, WordSegmenterModel}
import com.johnsnowlabs.nlp.annotator.BertForTokenClassification
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

val nerTagger = BertForTokenClassification
.pretrained("bert_token_classifier_ner_ud_gsd", "ja")
.setInputCols("sentence", "token")
.setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
sentence,
word_segmenter,
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
nlu.load("ja.classify.token_bert.classifier_ner_ud_gsd").predict("""Put your text here.""")
```

</div>

## Results

```bash
+--------------+--------+
|         token|     ner|
+--------------+--------+
|          宮本|B-PERSON|
|            茂|B-PERSON|
|            氏|       O|
|            は|       O|
|            、|       O|
|          日本|       O|
|            の|       O|
|          任天|       O|
|            堂|       O|
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
|Model Name:|bert_token_classifier_ner_ud_gsd|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|ja|
|Case sensitive:|true|
|Max sentense length:|128|

## Data Source

The model was trained on the Universal Dependencies, curated by Google. A NER version was created by megagonlabs:

https://github.com/megagonlabs/UD_Japanese-GSD

Reference:

Asahara, M., Kanayama, H., Tanaka, T., Miyao, Y., Uematsu, S., Mori, S., Matsumoto, Y., Omura, M., & Murawaki, Y. (2018). Universal Dependencies Version 2 for Japanese. In LREC-2018.

## Benchmarking

```bash
label  precision    recall  f1-score   support
DATE       0.87      0.94      0.91       191
EVENT       0.27      0.82      0.41        17
FAC       0.66      0.63      0.64        62
GPE       0.40      0.59      0.48        70
LANGUAGE       0.50      0.67      0.57         6
LAW       0.00      0.00      0.00         0
LOC       0.59      0.69      0.63        35
MONEY       0.90      0.90      0.90        20
MOVEMENT       0.27      1.00      0.43         3
NORP       0.46      0.57      0.50        46
O       0.98      0.97      0.98     11897
ORDINAL       0.94      0.70      0.80        43
ORG       0.49      0.43      0.46       204
PERCENT       1.00      0.80      0.89        20
PERSON       0.56      0.68      0.61       105
PRODUCT       0.32      0.53      0.40        30
QUANTITY       0.83      0.75      0.79       189
TIME       0.88      0.64      0.74        44
TITLE_AFFIX       0.33      0.80      0.47        10
WORK_OF_ART       0.67      0.76      0.71        42
accuracy          -         -      0.95     13034
macro-avg       0.60      0.69      0.62     13034
weighted-avg       0.96      0.95      0.95     13034

```
