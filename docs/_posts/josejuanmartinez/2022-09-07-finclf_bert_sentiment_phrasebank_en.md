---
layout: model
title: Financial Finbert Sentiment Analysis
author: John Snow Labs
name: finclf_bert_sentiment_phrasebank
date: 2022-09-07
tags: [en, finance, sentiment, classification, sentiment_analysis, licensed]
task: Sentiment Analysis
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the BERT language model in the finance domain, using a large financial corpus and thereby fine-tuning it for financial sentiment classification. Financial PhraseBank by Malo et al. (2014) and in-house JSL documents and annotations have been used for fine-tuning.

## Predicted Entities

`positive`, `negative`, `neutral`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN_FINANCE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_sentiment_phrasebank_en_1.0.0_3.2_1662539499618.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier_loaded = finance.BertForSequenceClassification.pretrained("finclf_bert_sentiment_phrasebank", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier_loaded    
])

# couple of simple examples
example = spark.createDataFrame([["Stocks rallied and the British pound gained."]]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

## Results

```bash
+--------------------+----------+
|                text|    result|
+--------------------+----------+
|Stocks rallied an...|[positive]|
+--------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_sentiment_phrasebank|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|409.9 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

In-house financial documents and Financial PhraseBank by Malo et al. (2014)

## Benchmarking

```bash
       label  precision    recall  f1-score   support
    positive       0.76      0.89      0.82       253
    negative       0.87      0.86      0.87       133
     neutral       0.94      0.87      0.90       584
    accuracy         -         -       0.87       970
   macro-avg       0.86      0.87      0.86       970
weighted-avg       0.88      0.87      0.88       970
```