---
layout: model
title: Detect Entities in General Scope (Few-NERD dataset)
author: John Snow Labs
name: nerdl_fewnerd_100d
date: 2021-07-02
tags: [ner, en, fewnerd, public, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.1.1
spark_version: 2.4
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained on Few-NERD/inter public dataset and it extracts 8 entities that are in general scope.

## Predicted Entities

`PERSON`, `ORGANIZATION`, `LOCATION`, `ART`, `BUILDING`, `PRODUCT`, `EVENT`, `OTHER`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_FEW_NERD/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_FewNERD.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_fewnerd_100d_en_3.1.1_2.4_1625227974733.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nerdl_fewnerd_100d_en_3.1.1_2.4_1625227974733.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

The model is trained using `glove_100d` word embeddings so, you should use the same embeddings in your nlp pipeline.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")\
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

ner = NerDLModel.pretrained("nerdl_fewnerd_100d") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('ner_chunk')

nlp_pipeline = Pipeline(stages=[document_assembler, sentencer, tokenizer, embeddings, ner, ner_converter])

l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = l_model.fullAnnotate("""The Double Down is a sandwich offered by Kentucky Fried Chicken (KFC) restaurants. He did not see active service again until 1882, when he took part in the Anglo-Egyptian War, and was present at the battle of Tell El Kebir (September 1882), for which he was mentioned in dispatches, received the Egypt Medal with clasp and the 3rd class of the Order of Medjidie, and was appointed a Companion of the Order of the Bath (CB).""")
```
```scala
...

val embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner = NerDLModel.pretrained("nerdl_fewnerd_100d")
.setInputCols(Array("sentence", "token", "embeddings")).setOutputCol("ner")

val ner_converter = NerConverter.setInputCols(Array("document", "token", "ner")) 
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner, ner_converter))
val data = Seq("The Double Down is a sandwich offered by Kentucky Fried Chicken (KFC) restaurants. He did not see active service again until 1882, when he took part in the Anglo-Egyptian War, and was present at the battle of Tell El Kebir (September 1882), for which he was mentioned in dispatches, received the Egypt Medal with clasp and the 3rd class of the Order of Medjidie, and was appointed a Companion of the Order of the Bath (CB).").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.fewnerd").predict("""The Double Down is a sandwich offered by Kentucky Fried Chicken (KFC) restaurants. He did not see active service again until 1882, when he took part in the Anglo-Egyptian War, and was present at the battle of Tell El Kebir (September 1882), for which he was mentioned in dispatches, received the Egypt Medal with clasp and the 3rd class of the Order of Medjidie, and was appointed a Companion of the Order of the Bath (CB).""")
```

</div>

## Results

```bash
+----------------------------------+---------+
|chunk                             |ner_label|
+----------------------------------+---------+
|Double Down                       |PRODUCT  |
|Kentucky Fried Chicken            |BUILDING |
|KFC                               |BUILDING |
|Anglo-Egyptian War                |EVENT    |
|Tell El Kebir                     |EVENT    |
|Egypt Medal                       |OTHER    |
|Order of Medjidie                 |OTHER    |
|Companion of the Order of the Bath|OTHER    |
+----------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_fewnerd_100d|
|Type:|ner|
|Compatibility:|Spark NLP 3.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Few-NERD:A Few-shot Named Entity Recognition Dataset, author: Ding, Ning and Xu, Guangwei and Chen, Yulin, and Wang, Xiaobin and Han, Xu and Xie, Pengjun and Zheng, Hai-Tao and Liu, Zhiyuan, book title: ACL-IJCNL, 2021.

## Benchmarking

```bash
+------------+-------+------+-------+-------+---------+------+------+
|      entity|     tp|    fp|     fn|  total|precision|recall|    f1|
+------------+-------+------+-------+-------+---------+------+------+
|      PERSON|21555.0|6194.0| 5643.0|27198.0|   0.7768|0.7925|0.7846|
|ORGANIZATION|36744.0|9059.0|13156.0|49900.0|   0.8022|0.7364|0.7679|
|    LOCATION|36367.0|7521.0| 7006.0|43373.0|   0.8286|0.8385|0.8335|
|         ART| 6170.0|1649.0| 2998.0| 9168.0|   0.7891| 0.673|0.7264|
|    BUILDING| 5112.0|2435.0| 3014.0| 8126.0|   0.6774|0.6291|0.6523|
|     PRODUCT| 8317.0|3253.0| 4325.0|12642.0|   0.7188|0.6579| 0.687|
|       OTHER|14461.0|4414.0| 5161.0|19622.0|   0.7661| 0.737|0.7513|
|       EVENT| 6024.0|1880.0| 2275.0| 8299.0|   0.7621|0.7259|0.7436|
+------------+-------+------+-------+-------+---------+------+------+

+------------------+
|             macro|
+------------------+
|0.7433252741184967|
+------------------+

+------------------+
|             micro|
+------------------+
|0.7703038245945377|
+------------------+
```