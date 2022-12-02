---
layout: model
title: Financial Sentiment Analysis
author: John Snow Labs
name: finclf_bert_sentiment
date: 2022-09-06
tags: [en, finance, sentiment, analysis, classification, licensed]
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

This model is a Sentiment Analysis fine-tuned model on 12K+ manually annotated (positive, negative, neutral) analyst reports on top of Financial Bert Embeddings. This model achieves superior performance on financial tone analysis task.

## Predicted Entities

`positive`, `negative`, `neutral`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN_FINANCE/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_sentiment_en_1.0.0_3.2_1662469460654.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = finance.BertForSequenceClassification.pretrained("finclf_bert_sentiment", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")
  
pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier  
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
|Stocks rallied an...|[Positive]|
+--------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_sentiment|
|Type:|finance|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|412.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

In-house annotations on financial reports

## Benchmarking

```bash
       label  precision    recall  f1-score   support
     neutral       0.91      0.87      0.89       588
    positive       0.76      0.81      0.78       251
    negative       0.83      0.87      0.85       131
    accuracy         -         -       0.86       970
   macro-avg       0.83      0.85      0.84       970
weighted-avg       0.86      0.86      0.86       970
```