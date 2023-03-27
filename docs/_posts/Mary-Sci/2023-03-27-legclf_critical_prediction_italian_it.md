---
layout: model
title: legclf_critical_prediction_italian
author: John Snow Labs
name: legclf_critical_prediction_italian
date: 2023-03-27
tags: [it, licensed, legal, classification, tensorflow]
task: Text Classification
language: it
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Description: This is a Binary classification model which identifies two criticality labels(critical, non-critical) in Italian-based Court Cases.

## Predicted Entities

`critical`, `non-critical`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_critical_prediction_italian_it_1.0.0_3.0_1679944691458.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_critical_prediction_italian_it_1.0.0_3.0_1679944691458.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

classifier = nlp.RoBertaForSequenceClassification.pretrained("legclf_critical_prediction_italian", "it", "legal/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

nlpPipeline = nlp.Pipeline(
      stages = [documentAssembler,
                tokenizer,
                classifier])
     
# Example text
example = spark.createDataFrame([["Per questi motivi, il Tribunale federale pronuncia: 1. Nella misura in cui è ammissibile, il ricorso è respinto. 2. Le spese giudiziarie di fr. 2'000.-- sono poste a carico del ricorrente. 3. Comunicazione ai patrocinatori delle parti, al patrocinatore di C._ e al Presidente della Camera di protezione del Tribunale d'appello del Cantone Ticino."]]).toDF("text")

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)

result = model.transform(example)

# result is a DataFrame
result.select("text", "class.result").show(truncate=100)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+----------+
|                                                                                                text|    result|
+----------------------------------------------------------------------------------------------------+----------+
|Per questi motivi, il Tribunale federale pronuncia: 1. Nella misura in cui è ammissibile, il rico...|[critical]|
+----------------------------------------------------------------------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_critical_prediction_italian|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|it|
|Size:|415.9 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Train dataset available [here](https://huggingface.co/datasets/rcds/legal_criticality_prediction)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
critical      0.82       0.90    0.86      10      
non_critical  0.95       0.91    0.93      23      
accuracy      -          -       0.91      33      
macro-avg     0.89       0.91    0.90      33      
weighted-avg  0.91       0.91    0.91      33    
```