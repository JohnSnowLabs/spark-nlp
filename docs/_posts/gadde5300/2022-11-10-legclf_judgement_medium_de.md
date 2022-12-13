---
layout: model
title: German Legal Judgement Classifier (Medium)
author: John Snow Labs
name: legclf_judgement_medium
date: 2022-11-10
tags: [de, legal, licensed, classification, judgement, german]
task: Text Classification
language: de
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `md` version of German Legal Judgement Text Classifier written in German legal writing style "Urteilsstil" (judgement style), which will retrieve if a text is either conclusion, definition, other or subsumption .

## Predicted Entities

`conclusion`, `definition`, `subsumption`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_judgement_medium_de_1.0.0_3.0_1668064600984.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_judgement_medium_de_1.0.0_3.0_1668064600984.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
document_assembler = nlp.DocumentAssembler() \
                .setInputCol("text") \
                .setOutputCol("document")
                
tokenizer = nlp.Tokenizer() \
                .setInputCols(["document"]) \
                .setOutputCol("token")
      
classifierdl = legal.BertForSequenceClassification.pretrained("legclf_judgement_medium","de", "legal/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("label")

bert_clf_pipeline = Pipeline(stages=[document_assembler,
                                     tokenizer,
                                     classifierdl])

text = ["Insoweit ergibt sich tatsächlich im Ergebnis ein Verzicht der Arbeitnehmer in Höhe der RoSi-Zulage ."]
empty_df = spark.createDataFrame([[""]]).toDF("text")
model = bert_clf_pipeline.fit(empty_df)
res = model.transform(spark.createDataFrame([text]).toDF("text"))

```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+-------------+
|text                                                                                                |result       |
+----------------------------------------------------------------------------------------------------+-------------+
|Insoweit ergibt sich tatsächlich im Ergebnis ein Verzicht der Arbeitnehmer in Höhe der RoSi-Zulage .|[subsumption]|
+----------------------------------------------------------------------------------------------------+-------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_judgement_medium|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|409.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

An in-house augmented version of [this dataset](https://zenodo.org/record/3936490#.Y2ybxctBxD-)

## Benchmarking

```bash

              precision    recall  f1-score   support

  conclusion       0.74      0.79      0.76       189
  definition       0.91      0.88      0.90       160
       other       0.85      0.82      0.83       163
 subsumption       0.71      0.70      0.70       159

    accuracy                           0.80       671
   macro avg       0.80      0.80      0.80       671
weighted avg       0.80      0.80      0.80       671

```
