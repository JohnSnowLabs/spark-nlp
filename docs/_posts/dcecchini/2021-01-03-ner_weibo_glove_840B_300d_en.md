---
layout: model
title: Named Entity Recognition for Chinese (GloVe 840B 300d)
author: John Snow Labs
name: ner_weibo_glove_840B_300d
date: 2021-01-03
tags: [en, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.

This model uses the pre-trained `glove_840B_300` embeddings model from `WordEmbeddings` annotator as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

|   Tag   | Meaning                            | Example      |
|:-------:|------------------------------------|--------------|
| PER.NAM | Person name                        | 张三         |
| PER.NOM | Code, category                     | 穷人         |
| LOC.NAM | Specific location                  | 紫玉山庄     |
| LOC.NOM | Generic location                   | 大峡谷、宾馆 |
| GPE.NAM | Administrative regions and areas   | 北京         |
| ORG.NAM | Specific organization              | 通惠医院     |
| ORG.NOM | Generic or collective organization | 文艺公司     |

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_ZH/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_weibo_glove_840B_300d_en_2.6.2_2.4_1609710175876.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
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

embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")\
          .setInputCols("document", "token") \
          .setOutputCol("embeddings")


ner = NerDLModel.pretrained("ner_weibo_glove_840B_300d", "zh") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")

pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        word_segmenter,
        embeddings,
        ner,
    ])

example = spark.createDataFrame(pd.DataFrame({'text': ["""张三去中国山东省泰安市爬中国五岳的泰山了"""]}))

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

val embeddings = WordEmbeddingsModel.pretrained("glove_840B_300", "xx")
          .setInputCols("document", "token")
          .setOutputCol("embeddings")


val ner = NerDLModel.pretrained("ner_weibo_glove_840B_300d", "zh")
        .setInputCols(["document", "token", "embeddings"])
        .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, embeddings, ner))

val result = pipeline.fit(Seq.empty["张三去中国山东省泰安市爬中国五岳的泰山了"].toDS.toDF("text")).transform(data)
```
</div>

## Results

```bash
+--------+-------+
|token   |ner    |
+--------+-------+
|张三    |PER.NAM|
|去      |O      |
|中国    |GPE.NAM|
|山东省  |GPE.NAM|
|泰安市  |O      |
|爬      |O      |
|中国五岳|GPE.NAM|
|的      |O      |
|泰山    |GPE.NAM|
|了      |O      |
+--------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_weibo_glove_840B_300d|
|Type:|ner|
|Compatibility:|Spark NLP 2.6.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

The model was trained on the [Weibo NER (He and Sun, 2017)](https://www.aclweb.org/anthology/E17-2113/) data set.

## Benchmarking

```bash
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| GPE.NAM      | 0.73      | 0.66   | 0.69     | 50      |
| GPE.NOM      | 0.00      | 0.00   | 0.00     | 2       |
| LOC.NAM      | 0.60      | 0.10   | 0.18     | 29      |
| LOC.NOM      | 0.20      | 0.10   | 0.13     | 10      |
| O            | 0.98      | 0.99   | 0.98     | 8605    |
| ORG.NAM      | 0.53      | 0.15   | 0.23     | 60      |
| ORG.NOM      | 0.50      | 0.28   | 0.36     | 18      |
| PER.NAM      | 0.66      | 0.61   | 0.63     | 139     |
| PER.NOM      | 0.68      | 0.63   | 0.66     | 197     |
| accuracy     | 0.96      | 9110   |          |         |
| macro avg    | 0.54      | 0.39   | 0.43     | 9110    |
| weighted avg | 0.96      | 0.96   | 0.96     | 9110    |
```
