---
layout: model
title: Legal Criticality Prediction Classifier in German
author: John Snow Labs
name: legclf_critical_prediction_legal
date: 2023-03-27
tags: [classification, de, licensed, legal, tensorflow]
task: Text Classification
language: de
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

This is a Binary classification model which identifies two criticality labels(critical, non-critical) in German-based Court Cases.

## Predicted Entities

`critical`, `non-critical`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_critical_prediction_legal_de_1.0.0_3.0_1679923757236.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_critical_prediction_legal_de_1.0.0_3.0_1679923757236.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

classifier = nlp.RoBertaForSequenceClassification.pretrained("legclf_critical_prediction_legal", "de", "legal/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

nlpPipeline = nlp.Pipeline(
      stages = [documentAssembler,
                tokenizer,
                classifier])
     
# Example text
example = spark.createDataFrame([["erkennt der Präsident: 1. Auf die Beschwerde wird nicht eingetreten. 2. Es werden keine Gerichtskosten erhoben. 3. Dieses Urteil wird den Parteien, dem Sozialversicherungsgericht des Kantons Zürich und dem Staatssekretariat für Wirtschaft (SECO) schriftlich mitgeteilt. Luzern, 5. Dezember 2016 Im Namen der I. sozialrechtlichen Abteilung des Schweizerischen Bundesgerichts Der Präsident: Maillard Der Gerichtsschreiber: Grünvogel"]]).toDF("text")

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)

result = model.transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+--------------+
|                                                                                                text|        result|
+----------------------------------------------------------------------------------------------------+--------------+
|erkennt der Präsident: 1. Auf die Beschwerde wird nicht eingetreten. 2. Es werden keine Gerichtsk...|[non_critical]|
+----------------------------------------------------------------------------------------------------+--------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_critical_prediction_legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|468.5 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Train dataset available [here](https://huggingface.co/datasets/rcds/legal_criticality_prediction)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
critical      0.76       0.87    0.81      249     
non_critical  0.87       0.76    0.81      293     
accuracy      -          -       0.81      542     
macro-avg     0.81       0.82    0.81      542     
weighted-avg  0.82       0.81    0.81      542     
```
