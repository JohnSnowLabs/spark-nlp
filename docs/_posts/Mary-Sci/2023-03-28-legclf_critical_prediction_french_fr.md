---
layout: model
title: Legal Criticality Prediction Classifier in French
author: John Snow Labs
name: legclf_critical_prediction_french
date: 2023-03-28
tags: [fr, licensed, classification, legal, tensorflow]
task: Text Classification
language: fr
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

This is a Binary classification model which identifies two criticality labels(critical, non-critical) in French-based Court Cases.

## Predicted Entities

`critical`, `non-critical`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_critical_prediction_french_fr_1.0.0_3.0_1680044769752.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_critical_prediction_french_fr_1.0.0_3.0_1680044769752.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

classifier = nlp.RoBertaForSequenceClassification.pretrained("legclf_critical_prediction_french", "fr", "legal/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

nlpPipeline = nlp.Pipeline(
      stages = [documentAssembler,
                tokenizer,
                classifier])
     
# Example text
example = spark.createDataFrame([["Par ces motifs, le Tribunal fédéral prononce : 1. Le recours est rejeté dans la mesure où il est recevable. 2. Les frais judiciaires, arrêtés à 2'000 fr., sont mis à la charge du recourant. 3. Le présent arrêt est communiqué au recourant, à la Commission du barreau ainsi qu'à la I e Cour administrative du Tribunal cantonal de l'Etat de Fribourg. Lausanne, le 19 janvier 2016 Au nom de la IIe Cour de droit public du Tribunal fédéral suisse Le Président : Zünd Le Greffier : Chatton"]]).toDF("text")

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)

result = model.transform(example)

# result is a DataFrame
result.select("text", "class.result").show(truncate=100)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+--------------+
|                                                                                                text|        result|
+----------------------------------------------------------------------------------------------------+--------------+
|Par ces motifs, le Tribunal fédéral prononce : 1. Le recours est rejeté dans la mesure où il est ...|[non_critical]|
+----------------------------------------------------------------------------------------------------+--------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_critical_prediction_french|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|415.9 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Train dataset available [here](https://huggingface.co/datasets/rcds/legal_criticality_prediction)

## Benchmarking

```bash
label         precision  recall  f1-score  support 
critical      0.74       0.74    0.74      117     
non_critical  0.81       0.81    0.81      161     
accuracy      -          -       0.78      278     
macro-avg     0.77       0.77    0.77      278     
weighted-avg  0.78       0.78    0.78      278     
```
