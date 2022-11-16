---
layout: model
title: Named Entity Recognition for Korean (GloVe 840B 300d)
author: John Snow Labs
name: ner_kmou_glove_840B_300d
date: 2021-01-03
task: Named Entity Recognition
language: ko
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [ko, ner, open_source]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations in the `BIO` format. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.

This model uses the pre-trained `glove_840B_300` embeddings model from `WordEmbeddings` annotator as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

Dates-`DT`, Locations-`LC`, Organizations-`OG`, Persons-`PS`, Time-`TI`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_KO/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_kmou_glove_840B_300d_ko_2.7.0_2.4_1609716021199.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

word_segmenter = WordSegmenterModel.pretrained("wordseg_kaist_ud", "ko")\
.setInputCols(["sentence"])\
.setOutputCol("token")
embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")\
.setInputCols("document", "token") \
.setOutputCol("embeddings")
ner = NerDLModel.pretrained("ner_kmou_glove_840B_300d", "ko") \
.setInputCols(["document", "token", "embeddings"]) \
.setOutputCol("ner")
...
pipeline = Pipeline(stages=[document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter])
example = spark.createDataFrame([['라이프니츠 의 주도 로 베를린 에 세우 어 지 ㄴ 베를린 과학아카데미']], ["text"])
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

val word_segmenter = WordSegmenterModel.pretrained("wordseg_kaist_ud", "ko")
.setInputCols(Array("sentence"))
.setOutputCol("token")
val embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")
.setInputCols(Array("document", "token"))
.setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_kmou_glove_840B_300d", "ko")
.setInputCols(Array("document", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter))
val data = Seq("라이프니츠 의 주도 로 베를린 에 세우 어 지 ㄴ 베를린 과학아카데미").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["라이프니츠 의 주도 로 베를린 에 세우 어 지 ㄴ 베를린 과학아카데미"]
ner_df = nlu.load('ko.ner').predict(text)
ner_df
```

</div>

## Results

```bash
+------------+----+
|token       |ner |
+------------+----+
|라이프니츠   |B-PS|
|베를린      |B-OG|
|과학아카데미  |I-OG|
+------------+----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_kmou_glove_840B_300d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ko|

## Data Source

The model was trained by the Korea Maritime and Ocean University [NLP data set](https://github.com/kmounlp/NER).

## Benchmarking

```bash
|    ner_tag   | precision | recall | f1-score | support |
|:------------:|:---------:|:------:|:--------:|:-------:|
|     B-DT     |    0.95   |  0.29  |   0.44   |   132   |
|     B-LC     |    0.00   |  0.00  |   0.00   |   166   |
|     B-OG     |    1.00   |  0.06  |   0.11   |   149   |
|     B-PS     |    0.86   |  0.13  |   0.23   |   287   |
|     B-TI     |    0.50   |  0.05  |   0.09   |    20   |
|     I-DT     |    0.94   |  0.36  |   0.52   |   164   |
|     I-LC     |    0.00   |  0.00  |   0.00   |    4    |
|     I-OG     |    1.00   |  0.08  |   0.15   |    25   |
|     I-PS     |    1.00   |  0.08  |   0.15   |    12   |
|     I-TI     |    0.50   |  0.10  |   0.17   |    10   |
|       O      |    0.94   |  1.00  |   0.97   |  12830  |
|   accuracy   |    0.94   |  13799 |          |         |
|   macro avg  |    0.70   |  0.20  |   0.26   |  13799  |
| weighted avg |    0.93   |  0.94  |   0.92   |  13799  |
```
