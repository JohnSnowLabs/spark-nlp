---
layout: model
title: Extract Tickers on Financial Texts
author: John Snow Labs
name: finner_ticker
date: 2022-08-09
tags: [en, financial, ner, ticker, trading, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model aims to detect Trading Symbols / Tickers in texts. You can then use Chunk Mappers to get more information about the company that ticker belongs to.

This is a light version of the model, trained on Tweets. You can find heavier models (transformer-based, more specifically RoBerta-based) in our Models Hub.

## Predicted Entities

`TICKER`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TICKER/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_ticker_en_1.0.0_3.2_1660037397073.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_ticker_en_1.0.0_3.2_1660037397073.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols(["document"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained("finner_ticker", "en", "finance/models")\
    .setInputCols(["document", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["document", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""TTSLA, DTV, AMZN, NFLX and GPRO continue to look good here. All ıf them need to continue and make it into"""]]).toDF("text")

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("ticker"),
                       F.expr("cols['1']['entity']").alias("label")).show()
```

</div>

## Results

```bash
+------+------+
|ticker| label|
+------+------+
| TTSLA|TICKER|
|   DTV|TICKER|
|  AMZN|TICKER|
|  NFLX|TICKER|
|  GPRO|TICKER|
+------+------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_ticker|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.2 MB|

## References

Original dataset (https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) and weak labelling on in-house texts

## Benchmarking

```bash
       label  precision    recall  f1-score   support
      TICKER       0.97      0.96      0.97      9823
   micro-avg       0.97      0.96      0.97      9823
   macro-avg       0.97      0.96      0.97      9823
weighted-avg       0.97      0.96      0.97      9823
```