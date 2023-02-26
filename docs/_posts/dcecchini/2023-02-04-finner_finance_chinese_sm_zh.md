---
layout: model
title: Chinese Financial NER (sm, bert_embeddings_mengzi_bert_base_fin)
author: John Snow Labs
name: finner_finance_chinese_sm
date: 2023-02-04
tags: [zh, cn, finance, ner, licensed]
task: Named Entity Recognition
language: zh
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is the small version of the NER model for Financial Chinese texts, trained in a subset of **ChFinAnn** (see "Datasets used for training"). 

To use this model, use the `BertEmbeddings` model named `bert_embeddings_mengzi_bert_base_fin"` as:

```python
bert_embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_mengzi_bert_base_fin","zh") \
    .setInputCols("sentence", "token") \
    .setOutputCol("embeddings")
```

Also, please note that the Chinese texts are not separated by white space. The embedding model we use is based on character-level embeddings, so you need to split the text on every character (for example, by setting `.setSplitPattern("")` in the `Tokenizer` annotator).

## Predicted Entities

`AveragePrice`, `ClosingDate`, `CompanyName`, `EndDate`, `EquityHolder`, `FrozeShares`, `HighestTradingPrice`, `LaterHoldingShares`, `LegalInstitution`, `LowestTradingPrice`, `OtherType`, `PledgedShares`, `Pledgee`, `ReleasedDate`, `RepurchaseAmount`, `RepurchasedShares`, `StartDate`, `StockAbbr`, `StockCode`, `TotalHoldingRatio`, `TotalHoldingShares`, `TotalPledgedShares`, `TradedShares`, `UnfrozeDate`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_FINANCE_CHINESE){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_finance_chinese_sm_zh_1.0.0_3.0_1675554138686.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_finance_chinese_sm_zh_1.0.0_3.0_1675554138686.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
tokenizer = nlp.Tokenizer()\
        .setInputCols(["document"])\
        .setOutputCol("token")\
        .setSplitPattern("") # Split on char level

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_mengzi_bert_base_fin","en") \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_finance_chinese_sm", "zh", "finance/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = ["""近日，渤海水业股份有限公司（以下简称“公司”）收到公司持股5%以上股东李华青女士的《告知函》，获悉李华青女士将其所持有的部分公司股票进行补充质押，具体事项如下："""]

res = model. Transform(spark.createDataFrame([text]).toDF("text"))
res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"),
              F.expr("cols['1']['confidence']").alias("confidence")).show(truncate=False)

```

</div>

## Results

```bash
+------------------------------------+------------+----------+
|chunk                               |ner_label   |confidence|
+------------------------------------+------------+----------+
|业股份有限公司（以下简称“公司”）收到|CompanyName |0.7933    |
|质押，具体                          |EquityHolder|0.9378667 |
+------------------------------------+------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_finance_chinese_sm|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|zh|
|Size:|16.8 MB|

## References

The dataset used for training was a subset of the **chFinAnn** dataset, consisting of financial statements of Chinese listed companies from 2008 to 2018. 

Reference:

- [Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction](https://aclanthology.org/D19-1032) (Zheng et al., EMNLP-IJCNLP 2019)

## Sample text from the training dataset

近日，渤海水业股份有限公司（以下简称“公司”）收到公司持股5%以上股东李华青女士的《告知函》，获悉李华青女士将其所持有的部分公司股票进行补充质押，具体事项如下：

## Benchmarking

```bash
 entity                 precision    recall        f1    support 
 AveragePrice            78.0731   85.1449   81.4558         301 
 ClosingDate             76.0148   57.7031   65.6051         271 
 CompanyName             94.0767   95.9251   94.9919        5605 
 EndDate                 75.487    44.5402   56.0241         616 
 EquityHolder            83.8303   91.319    87.4146        7780 
 FrozeShares             47.4227   31.7241   38.0165          97 
 HighestTradingPrice     74.4186   70.5085   72.4108         559 
 LaterHoldingShares      31.4961   11.9048   17.2786         127 
 LegalInstitution        92.3767   87.6596   89.9563         223 
 LowestTradingPrice      78.5047   52.1739   62.6866         107 
 OtherType               78.2961   36.7619   50.0324         493 
 PledgedShares           78.0776   65.5189   71.249         1779 
 Pledgee                 90.1003   88.656    89.3723        1596 
 ReleasedDate            54.5016   46.1853   50              622 
 RepurchaseAmount        68.323    72.8477   70.5128         322 
 RepurchasedShares       79.5918   70.4819   74.7604         588 
 StartDate               65.4217   77.5493   70.9711        4150 
 StockAbbr               83.7656   82.0521   82.9           2969 
 StockCode               99.8355   99.5626   99.6989        1824 
 TotalHoldingRatio       74.2574   85.1628   79.3371        1515 
 TotalHoldingShares      67.0582   88.7306   76.387         2043 
 TotalPledgedShares      74.5989   86.0226   79.9045        1122 
 TradedShares            76.9231   68.6948   72.5765         910 
 UnfrozeDate              5.88235   5.55556   5.71429         17
```