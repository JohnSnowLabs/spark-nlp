---
layout: model
title: Legal ORG, PRODUCT and ALIAS NER
author: John Snow Labs
name: legner_orgs_prods_alias
date: 2022-08-17
tags: [en, legal, ner, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a large Named Entity Recognition model, trained with a subset of generic conLL, financial and legal conll, ontonotes and several in-house corpora, to detect Organizations, Products and Aliases of Companies.

## Predicted Entities

`ORG`, `PROD`, `ALIAS`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_ORGPROD){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_orgs_prods_alias_en_1.0.0_3.2_1660733903868.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_orgs_prods_alias_en_1.0.0_3.2_1660733903868.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained("legner_orgs_prods_alias","en","legal/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""This INTELLECTUAL PROPERTY AGREEMENT (this "Agreement"), dated as of December 31, 2018 (the "Effective Date") is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ("Seller") and AFI Licensing LLC, a Delaware limited liability company ("Licensing" and together with Seller, "Arizona") and AHF Holding, Inc. (formerly known as Tarzan HoldCo, Inc.), a Delaware corporation ("Buyer") and Armstrong Hardwood Flooring Company, a Tennessee corporation (the "Company" and together with Buyer the "Buyer Entities") (each of Arizona on the one hand and the Buyer Entities on the other hand, a "Party" and collectively, the "Parties").
"""]

res = model.transform(spark.createDataFrame([text]).toDF("text"))
res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"),
              F.expr("cols['1']['confidence']").alias("confidence")).show(truncate=False)
```

</div>

## Results

```bash
+-----------------------------------+---------+----------+
|chunk                              |ner_label|confidence|
+-----------------------------------+---------+----------+
|Armstrong Flooring, Inc            |ORG      |0.807575  |
|Seller                             |ALIAS    |0.997     |
|AFI Licensing LLC                  |ORG      |0.7076333 |
|Licensing                          |ALIAS    |0.9981    |
|Seller                             |ALIAS    |0.996     |
|Arizona                            |ALIAS    |0.9958    |
|AHF Holding, Inc.                  |ORG      |0.72438   |
|Tarzan HoldCo, Inc                 |ORG      |0.684675  |
|Buyer                              |ALIAS    |0.9983    |
|Armstrong Hardwood Flooring Company|ORG      |0.58274996|
|Company                            |ALIAS    |0.9989    |
|Buyer                              |ALIAS    |0.9979    |
|Buyer Entities                     |ALIAS    |0.98835003|
|Arizona                            |ALIAS    |0.9635    |
|Buyer Entities                     |ALIAS    |0.77565   |
|Party                              |ALIAS    |0.9982    |
+-----------------------------------+---------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_orgs_prods_alias|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.7 MB|

## References

ConLL-2003, FinSec ConLL, a subset of Ontonotes, In-house corpora

## Benchmarking

```bash
label          tp        fp      fn      prec          rec           f1
I-ORG          12853	 2621    2685    0.8306191     0.82719785    0.828905
B-PRODUCT      2306      697     932     0.76789874    0.712168      0.7389841
I-ALIAS        14        6       13      0.7           0.5185185     0.59574467
B-ORG          8967      2078    2311    0.81186056    0.79508775    0.80338657
I-PRODUCT      2336      803     1091    0.74418604    0.68164575    0.7115443
B-ALIAS        76        14      22      0.84444445    0.7755102     0.80851066
Macro-average  26552     6219    7054    0.78316814    0.7183547     0.7493626
Micro-average  26552     6219    7054    0.8102285     0.790097      0.80003613
```



