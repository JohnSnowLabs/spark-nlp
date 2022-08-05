---
layout: model
title: Financial Finbert Sentiment Analysis
author: John Snow Labs
name: finance_finbert_sentiment_analysis
date: 2022-08-05
tags: [en, finance, sentiment, classification, sentiment_analysis, licensed]
task: Text Classification
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
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
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finance_finbert_sentiment_analysis_en_1.0.0_3.2_1659709126694.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("finance_finbert_sentiment_analysis", "en", "finance/models")\
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
|Model Name:|finance_finbert_sentiment_analysis|
|Type:|finance|
|Compatibility:|Spark NLP for Finance 1.0.0+|
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
              precision    recall  f1-score   support

    positive       0.76      0.89      0.82       253
    negative       0.87      0.86      0.87       133
     neutral       0.94      0.87      0.90       584

    accuracy                           0.87       970
   macro avg       0.86      0.87      0.86       970
weighted avg       0.88      0.87      0.88       970
```