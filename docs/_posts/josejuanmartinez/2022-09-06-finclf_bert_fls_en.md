---
layout: model
title: Forward-Looking Statements Classification
author: John Snow Labs
name: finclf_bert_fls
date: 2022-09-06
tags: [en, finance, forward, looking, statements, fls, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Text Classification model aimed to detect at sentence or paragraph level, if there is a Forward-looking statements (FLS).

FLS are beliefs and opinions about firm's future events or results, usually present in documents as Financial Reports. Identifying forward-looking statements from corporate reports can assist investors in financial analysis. 

This model was trained originally on 3,500 manually annotated sentences from Management Discussion and Analysis section of annual reports of Russell 3000 firms and then finetuned in house by JSL on low-performant examples.

## Predicted Entities

`Specific FLS`, `Non-specific FLS`, `Not FLS`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/FINCLF_FLS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_fls_en_1.0.0_3.2_1662468990598.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = finance.BertForSequenceClassification.pretrained("finclf_bert_fls", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")
  
pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier
])

# couple of simple examples
example = spark.createDataFrame([["Global economy will increase during the next year."]]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

## Results

```bash
+--------------------+--------------+
|                text|        result|
+--------------------+--------------+
|Global economy wi...|[Specific FLS]|
+--------------------+--------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_fls|
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

In-house annotations on 10K financial reports and reports from Russell 3000 firms

## Benchmarking

```bash
           label  precision    recall  f1-score   support
    Specific_FLS       0.96      0.93      0.94       311
Non-specific_FLS       0.91      0.94      0.92       215
         Not_FLS       0.84      0.87      0.85        70
        accuracy          -         -      0.92       596
       macro-avg       0.90      0.91      0.91       596
    weighted-avg       0.93      0.92      0.92       596
```
