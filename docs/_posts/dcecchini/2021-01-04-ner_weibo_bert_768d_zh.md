---
layout: model
title: Named Entity Recognition for Chinese (BERT-Weibo Dataset)
author: John Snow Labs
name: ner_weibo_bert_768d
date: 2021-01-04
task: Named Entity Recognition
language: zh
edition: Spark NLP 2.7.0
spark_version: 2.4
tags: [zh, ner, open_source]
supported: true
annotator: NerDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.

This model uses the pre-trained `bert_base_chinese` embeddings model from `BertEmbeddings` annotator as an input, so be sure to use the same embeddings in the pipeline.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_weibo_bert_768d_zh_2.7.0_2.4_1609719542498.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")\
        .setInputCols(["sentence"])\
        .setOutputCol("token")
embeddings = BertEmbeddings.pretrained(name='bert_base_chinese', lang='zh')\
          .setInputCols("document", "token") \
          .setOutputCol("embeddings")
ner = NerDLModel.pretrained("ner_weibo_bert_768d", "zh") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
...
pipeline = Pipeline(stages=[document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter])
example = spark.createDataFrame([['张三去中国山东省泰安市爬中国五岳的泰山了']], ["text"])
result = pipeline.fit(example).transform(example)
```

```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
        
val sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")


val word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")
     .setInputCols(Array("sentence"))
     .setOutputCol("token")
val embeddings = BertEmbeddings.pretrained(name='bert_base_chinese', lang='zh')
     .setInputCols(Array("document", "token"))
     .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_weibo_bert_768d", "zh")
     .setInputCols(Array("document", "token", "embeddings"))
     .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, embeddings, ner))
val data = Seq("张三去中国山东省泰安市爬中国五岳的泰山了").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
text = ["张三去中国山东省泰安市爬中国五岳的泰山了"]

ner_df = nlu.load('zh.ner.weibo.bert_768d').predict(text)
ner_df
```

</div>

## Results

```bash
+--------+-------+
|token   |ner    |
+--------+-------+
|张三    |PER.NAM|
|中国    |GPE.NAM|
|山东省  |GPE.NAM|
|中国五岳|GPE.NAM|
|泰山    |GPE.NAM|
+--------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_weibo_bert_768d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|zh|

## Data Source

The model was trained on the [Weibo NER (He and Sun, 2017)](https://www.aclweb.org/anthology/E17-2113/) data set.

## Benchmarking

```bash
| ner_tag      | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| GPE.NAM      | 0.73      | 0.66   | 0.69     | 50      |
| GPE.NOM      | 0.00      | 0.00   | 0.00     | 2       |
| LOC.NAM      | 0.60      | 0.10   | 0.18     | 29      |
| LOC.NOM      | 0.20      | 0.10   | 0.13     | 10      |
| ORG.NAM      | 0.53      | 0.15   | 0.23     | 60      |
| ORG.NOM      | 0.50      | 0.28   | 0.36     | 18      |
| PER.NAM      | 0.66      | 0.61   | 0.63     | 139     |
| PER.NOM      | 0.68      | 0.63   | 0.66     | 197     |
| accuracy     | 0.96      | 9110   |          |         |
| macro avg    | 0.54      | 0.39   | 0.43     | 9110    |
| weighted avg | 0.96      | 0.96   | 0.96     | 9110    |
```
