---
layout: model
title: Financial Executives Item Binary Classifier
author: John Snow Labs
name: finclf_executives_item
date: 2022-08-10
tags: [en, finance, classification, 10k, annual, reports, sec, filings, licensed]
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

This model is a Binary Classifier (True, False) for the `executives` item type of 10K Annual Reports. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

If you have big financial documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Finance NLP Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Finance/1.Tokenization_Splitting.ipynb)), namely:
- Paragraph splitting (by multiline);
- Splitting by headers / subheaders;
- etc.

Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

## Predicted Entities

`other`, `executives`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_executives_item_en_1.0.0_3.2_1660154397820.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
     .setInputCol("text") \
     .setOutputCol("document")

useEmbeddings = nlp.UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

docClassifier = nlp.ClassifierDLModel.pretrained("finclf_executives_item", "en", "finance/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    useEmbeddings,
    docClassifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

## Results

```bash
+-------+
| result|
+-------+
|[executives]|
|[other]|
|[other]|
|[executives]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_executives_item|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|22.6 MB|

## References

Weak labelling on documents from Edgar database

## Benchmarking

```bash
       label  precision    recall  f1-score   support
  executives       0.96      0.98      0.97        46
       other       0.98      0.96      0.97        45
    accuracy        -         -        0.97        91
   macro-avg       0.97      0.97      0.97        91
weighted-avg       0.97      0.97      0.97        91
```