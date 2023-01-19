---
layout: model
title: Extract Tickers on Financial Texts (RoBerta)
author: John Snow Labs
name: finner_roberta_ticker
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

This model aims to detect Trading Symbols / Tickers in texts. You can then use Chunk Mappers to get more information about the company that ticker belongs to. This is a RoBerta-based model, you can find other lighter versions of this model in Models Hub.

## Predicted Entities

`TICKER`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TICKER/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_roberta_ticker_en_1.0.0_3.2_1660036613729.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_roberta_ticker_en_1.0.0_3.2_1660036613729.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
      .setInputCol('text')\
      .setOutputCol('document')

tokenizer = nlp.Tokenizer()\
      .setInputCols("document")\
      .setOutputCol("token")

tokenClassifier = nlp.RoBertaForTokenClassification.pretrained("finner_roberta_ticker", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
      .setInputCols(["document", "token", "ner"])\
      .setOutputCol("ner_chunk")

pipeline = Pipeline().setStages([document_assembler,
                                 tokenizer, 
                                 tokenClassifier,
                                 ner_converter])

text = ["""There are some serious purchases and sales of AMZN stock today."""]

test_data = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(test_data)
res= model.transform(test_data)

res.select('ner_chunk').collect()
```

</div>

## Results

```bash
['AMZN']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_roberta_ticker|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|465.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

Original dataset (https://www.kaggle.com/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) and weak labelling on in-house texts

## Benchmarking

```bash
       label  precision    recall  f1-score   support
      TICKER       0.98      0.97      0.98      9823
   micro-avg       0.98      0.97      0.98      9823
   macro-avg       0.98      0.97      0.98      9823
weighted-avg       0.98      0.97      0.98      9823
```
