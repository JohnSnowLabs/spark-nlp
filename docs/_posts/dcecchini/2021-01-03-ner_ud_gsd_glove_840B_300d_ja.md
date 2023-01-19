---
layout: model
title: Named Entity Recognition for Japanese (GloVe 840B 300d)
author: John Snow Labs
name: ner_ud_gsd_glove_840B_300d
date: 2021-01-03
task: Named Entity Recognition
language: ja
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [ner, ja, open_source]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.

This model uses the pre-trained `glove_840B_300` embeddings model from `WordEmbeddings` annotator as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

`DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `MOVEMENT`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `TITLE_AFFIX`,  and `WORK_OF_ART`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_JA/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_ud_gsd_glove_840B_300d_ja_2.7.0_2.4_1609712569080.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ner_ud_gsd_glove_840B_300d_ja_2.7.0_2.4_1609712569080.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")\
.setInputCols("document", "token") \
.setOutputCol("embeddings")
ner = NerDLModel.pretrained("ner_ud_gsd_glove_840B_300d", "ja") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("entities")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter])
example = spark.createDataFrame([['5月13日に放送されるフジテレビ系「僕らの音楽」にて、福原美穂とAIという豪華共演が決定した。']], ["text"])
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_gsd_ud", "ja")
.setInputCols(Array("sentence"))
.setOutputCol("token")
val embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")
.setInputCols(Array("document", "token"))
.setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_ud_gsd_glove_840B_300d", "ja")
.setInputCols(Array("document", "token", "embeddings"))
.setOutputCol("ner")
val ner_converter = new NerConverter()
  .setInputCols("sentence", "token", "ner")
  .setOutputCol("entities")
  
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter))
val data = Seq("5月13日に放送されるフジテレビ系「僕らの音楽」にて、福原美穂とAIという豪華共演が決定した。").toDF("text")
val result = pipeline.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("ja.ner").predict("""Put your text here.""")
```

</div>

## Results

```bash
+----------+------+
|token     |ner   |
+----------+------+
|5月       |DATE  |
|13日      |DATE  |
|に        |O     |
|放送      |O     |
|さ        |O     |
|れる      |O     |
|フジテレビ|O     |
|系        |O     |
|「        |O     |
|僕らの音楽|O     |
|」        |O     |
|にて      |O     |
|、        |O     |
|福原美穂  |PERSON|
|と        |O     |
|AI        |O     |
|と        |O     |
|いう      |O     |
|豪華      |O     |
|共演      |O     |
|が        |O     |
|決定      |O     |
|し        |O     |
|た        |O     |
|。        |O     |
+----------+------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ud_gsd_glove_840B_300d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ja|

## Data Source

The model was trained on the Universal Dependencies, curated by Google.

Reference:

> Asahara, M., Kanayama, H., Tanaka, T., Miyao, Y., Uematsu, S., Mori, S., Matsumoto, Y., Omura, M., & Murawaki, Y. (2018). Universal Dependencies Version 2 for Japanese. In LREC-2018.

## Benchmarking

```bash
|    ner_tag   | precision | recall | f1-score | support |
|:------------:|:---------:|:------:|:--------:|:-------:|
|     DATE     |    1.00   |  0.86  |   0.92   |    84   |
|     EVENT    |    1.00   |  0.14  |   0.25   |    14   |
|      FAC     |    1.00   |  0.15  |   0.26   |    20   |
|      GPE     |    1.00   |  0.01  |   0.02   |    82   |
|   LANGUAGE   |    0.00   |  0.00  |   0.00   |    6    |
|      LAW     |    0.00   |  0.00  |   0.00   |    3    |
|      LOC     |    0.00   |  0.00  |   0.00   |    25   |
|     MONEY    |    0.86   |  0.86  |   0.86   |    7    |
|   MOVEMENT   |    0.00   |  0.00  |   0.00   |    4    |
|     NORP     |    1.00   |  0.11  |   0.19   |    28   |
|    ORDINAL   |    0.92   |  0.85  |   0.88   |    13   |
|      ORG     |    0.44   |  0.35  |   0.39   |    75   |
|    PERCENT   |    1.00   |  1.00  |   1.00   |    7    |
|    PERSON    |    0.71   |  0.06  |   0.10   |    89   |
|    PRODUCT   |    0.42   |  0.48  |   0.45   |    23   |
|   QUANTITY   |    0.98   |  0.78  |   0.87   |    78   |
|     TIME     |    1.00   |  1.00  |   1.00   |    13   |
|  TITLE_AFFIX |    0.00   |  0.00  |   0.00   |    20   |
|  WORK_OF_ART |    1.00   |  0.22  |   0.36   |    18   |
|   accuracy   |    0.97   |  12419 |          |         |
|   macro avg  |    0.67   |  0.39  |   0.43   |  12419  |
| weighted avg |    0.96   |  0.97  |   0.96   |  12419  |
```
