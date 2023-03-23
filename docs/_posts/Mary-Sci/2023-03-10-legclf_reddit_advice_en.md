---
layout: model
title: Legal Advice Class Identifier
author: John Snow Labs
name: legclf_reddit_advice
date: 2023-03-10
tags: [en, licensed, legal, classifier, reddit, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
engine: tensorflow
annotator: MedicalBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multiclass classification model which retrieves the topic/class of an informal message from a legal forum, including the following classes: `digital`, `business`, `insurance`, `contract`, `driving`, `school`, `family`, `wills`, `employment`, `housing`, `criminal`.

## Predicted Entities

`digital`, `business`, `insurance`, `contract`, `driving`, `school`, `family`, `wills`, `employment`, `housing`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_reddit_advice_en_1.0.0_3.0_1678448985639.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_reddit_advice_en_1.0.0_3.0_1678448985639.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

seq_classifier = legal.BertForSequenceClassification.pretrained("legclf_reddit_advice", "en", "legal/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = nlp.Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["Mother of my child took my daughter and moved (without notice), won't let me see her or tell me where she is."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+--------+
|  result|
+--------+
|[family]|
+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_reddit_advice|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.4 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Train dataset available [here](https://huggingface.co/datasets/jonathanli/legal-advice-reddit)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
business      0.76       0.67    0.72      239     
contract      0.80       0.68    0.73      207     
criminal      0.82       0.77    0.80      209     
digital       0.76       0.74    0.75      223     
driving       0.86       0.85    0.86      223     
employment    0.76       0.92    0.83      222     
family        0.88       0.95    0.92      216     
housing       0.89       0.95    0.92      221     
insurance     0.83       0.80    0.81      221     
school        0.87       0.91    0.89      207     
wills         0.95       0.96    0.96      199     
accuracy      -          -       0.83      2387    
macro-avg     0.84       0.84    0.83      2387    
weighted-avg  0.83       0.83    0.83      2387
```
