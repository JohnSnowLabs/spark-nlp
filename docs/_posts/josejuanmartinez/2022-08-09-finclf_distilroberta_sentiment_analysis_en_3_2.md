---
layout: model
title: Financial Finbert Sentiment Analysis (DistilRoBerta)
author: John Snow Labs
name: finclf_distilroberta_sentiment_analysis
date: 2022-08-09
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

This model is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the DistilRoBerta language model in the finance domain, using a financial corpus and thereby fine-tuning it for financial sentiment classification. Financial PhraseBank by Malo et al. (2014) and in-house JSL documents and annotations have been used for fine-tuning.

## Predicted Entities

`positive`, `negative`, `neutral`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN_FINANCE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_distilroberta_sentiment_analysis_en_1.0.0_3.2_1660055192412.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finclf_distilroberta_sentiment_analysis_en_1.0.0_3.2_1660055192412.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

classifier = nlp.RoBertaForSequenceClassification.pretrained("finclf_distilroberta_sentiment_analysis","en", "finance/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")


nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          tokenizer,
          classifier])
    

# couple of simple examples
example = spark.createDataFrame([["Stocks rallied and the British pound gained."]]).toDF("text")

result = nlpPipeline.fit(example).transform(example)

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
|Model Name:|finclf_distilroberta_sentiment_analysis|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|309.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

In-house financial documents and Financial PhraseBank by Malo et al. (2014)

## Benchmarking

```bash
       label  precision    recall  f1-score   support
    positive       0.77      0.88      0.81       253
    negative       0.86      0.85      0.88       133
     neutral       0.93      0.86      0.90       584
    accuracy         -         -       0.86       970
   macro-avg       0.85      0.86      0.85       970
weighted-avg       0.87      0.86      0.87       970
```
