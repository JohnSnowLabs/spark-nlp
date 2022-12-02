---
layout: model
title: Spanish NER for Laws and Money
author: John Snow Labs
name: legner_law_money
date: 2022-09-28
tags: [es, legal, ner, laws, money, licensed]
task: Named Entity Recognition
language: es
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Spanish Named Entity Recognition model for detecting laws and monetary ammounts. This model was trained in-house and available annotations of this [dataset](https://huggingface.co/datasets/scjnugacj/scjn_dataset_ner) and weak labelling from this [model](https://huggingface.co/pitoneros/NER_LAW_MONEY4)

## Predicted Entities

`LAW`, `MONEY`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_law_money_es_1.0.0_3.0_1664362333282.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
.setInputCols("sentence") \
.setOutputCol("token")

tokenClassifier = nlp.RoBertaForTokenClassification.pretrained("legner_law_money", "es", "legal/models") \
.setInputCols(["sentence", "token"]) \
.setOutputCol("ner")

pipeline = Pipeline(
    stages=[
      documentAssembler, 
      sentenceDetector, 
      tokenizer, 
      tokenClassifier])

text = "La recaudación del ministerio del interior fue de 20,000,000 euros así constatado por el artículo 24 de la Constitución Española."

data = spark.createDataFrame([[""]]).toDF("text")

fitmodel = pipeline.fit(data)

light_model = LightPipeline(fitmodel)

light_result = light_model.fullAnnotate(text)

chunks = []
entities = []

for n in light_result[0]['ner_chunk']:       
    print("{n.result} ({n.metadata['entity']}))
```

</div>

## Results

```bash
20,000,000 euros (MONEY)
artículo 24 de la Constitución Española (LAW)
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_law_money|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|414.2 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

This model was trained in-house and available annotations of this [dataset](https://huggingface.co/datasets/scjnugacj/scjn_dataset_ner) and weak labelling from this [model](https://huggingface.co/pitoneros/NER_LAW_MONEY4)

## Benchmarking

```bash
           label  precision    recall  f1-score   support
             LAW       0.95      0.96      0.96        20
           MONEY       0.98      0.99      0.99       106
        accuracy         -         -       0.98       126
       macro-avg       0.97      0.98      0.97       126
    weighted-avg       0.98      0.99      0.99       126
```