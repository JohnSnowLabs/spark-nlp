---
layout: model
title: Named Entity Recognition for Chinese (BERT-MSRA Dataset)
author: John Snow Labs
name: ner_msra_bert_768d
date: 2021-01-03
tags: [zh, cn, ner, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model annotates named entities in a text, that can be used to find features such as names of people, places, and organizations. The model does not read words directly but instead reads word embeddings, which represent words as points such that more semantically similar words are closer together.

This model uses the pre-trained `bert_base_chinese` embeddings model from `BertEmbeddings` annotator as an input, so be sure to use the same embeddings in the pipeline.

## Predicted Entities

Persons-`PER`, Locations-`LOC`, Organizations-`ORG`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_ZH/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ner_msra_bert_768d_zh_2.7.0_2.4_1609703549977.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```python
...
word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")\
        .setInputCols(["sentence"])\
        .setOutputCol("token")
embeddings = BertEmbeddings.pretrained(name='bert_base_chinese', lang='zh')\
          .setInputCols("document", "token") \
          .setOutputCol("embeddings")
ner = NerDLModel.pretrained("ner_msra_bert_768d", "zh") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
...
pipeline = Pipeline(stages=[document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter])
example = spark.createDataFrame(pd.DataFrame({'text': ["""马云在浙江省杭州市出生，是阿里巴巴集团的主要创始人。"""]}))
result = pipeline.fit(example).transform(example)
```
```scala
...
val word_segmenter = WordSegmenterModel.pretrained("wordseg_large", "zh")
        .setInputCols(Array("sentence"))
        .setOutputCol("token")
val embeddings = BertEmbeddings.pretrained(name='bert_base_chinese', lang='zh')
        .setInputCols(Array("document", "token"))
        .setOutputCol("embeddings")
val ner = NerDLModel.pretrained("ner_msra_bert_768d", "zh")
        .setInputCols(["document", "token", "embeddings"])
        .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, word_segmenter, embeddings, ner, ner_converter))
val result = pipeline.fit(Seq.empty["马云在浙江省杭州市出生，是阿里巴巴集团的主要创始人。"].toDS.toDF("text")).transform(data)
```

</div>

## Results

```bash
+------------+---+
|token       |ner|
+------------+---+
|马云        |PER|
|在          |O  |
|浙江省      |LOC|
|杭州市      |LOC|
|出生        |ORG|
|，          |O  |
|是          |O  |
|阿里巴巴集团|ORG|
|的          |O  |
|主要        |O  |
|创始人      |PER|
|。          |O  |
+------------+---+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_msra_bert_768d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|zh|

## Data Source

The model was trained on the [MSRA (Levow, 2006)](https://www.aclweb.org/anthology/W06-0115/) data set created by "Microsoft Research Asia".

## Benchmarking

```bash
|      ner_tag        | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| LOC          | 0.97      | 0.97   | 0.97     | 2777    |
| O            | 1.00      | 1.00   | 1.00     | 146826  |
| ORG          | 0.88      | 0.99   | 0.93     | 1292    |
| PER          | 0.97      | 0.97   | 0.97     | 1430    |
| accuracy     | 1.00      | 152325 |          |         |
| macro avg    | 0.95      | 0.98   | 0.97     | 152325  |
| weighted avg | 1.00      | 1.00   | 1.00     | 152325  |
```
