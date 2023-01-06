---
layout: model
title: Legal Sick Days Clause Binary Classifier
author: John Snow Labs
name: legclf_sick_days_clause
date: 2022-12-05
tags: [en, legal, sick_days, classification, clauses, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the `sick-days` clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

        If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Legal NLP Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Legal/1.Tokenization_Splitting.ipynb)), namely:
        - Paragraph splitting (by multiline);
        - Splitting by headers / subheaders;
        - etc.

        Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

        This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

## Predicted Entities

`sick-days`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_sick_days_clause_en_1.0.0_3.0_1670246331980.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
 documentAssembler = nlp.DocumentAssembler() \
         .setInputCol("clause_text") \
         .setOutputCol("document")

embeddings = nlp.BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings") 

docClassifier = legal.ClassifierDLModel.pretrained("legclf_sick_days_clause", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")

nlpPipeline = nlp.Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])

df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("clause_text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_sick_days_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.7 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house

## Benchmarking

```bash
       label  precision    recall  f1-score   support
       other       1.00      1.00      1.00        39
   sick-days       1.00      1.00      1.00        22
    accuracy          -         -      1.00        61
   macro-avg       1.00      1.00      1.00        61
weighted-avg       1.00      1.00      1.00        61
```