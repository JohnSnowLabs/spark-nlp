---
layout: model
title: Legal Termination Clause Binary Classifier (CUAD dataset, SBERT version)
author: John Snow Labs
name: legclf_sbert_cuad_termination_clause
date: 2022-11-11
tags: [termination, en, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: ClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the `termination` clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

This version was trained with Universal Sentence Encoder. There is another version using Universal Sentence Encoding, called `legclf_cuad_termination_clause`

If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Spark NLP for Legal Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Legal/1.Tokenization_Splitting.ipynb)), namely:
- Paragraph splitting (by multiline);
- Splitting by headers / subheaders;
- etc.

Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

There are other models in this dataset with similar title, but the difference is the dataset it was trained on. This one was trained with `cuad` dataset.

## Predicted Entities

`termination`, `other`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/CLASSIFY_LEGAL_CLAUSES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_sbert_cuad_termination_clause_en_1.0.0_3.0_1668163200458.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler() \
     .setInputCol("clause_text") \
     .setOutputCol("document")
  
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

docClassifier = nlp.ClassifierDLModel.pretrained("legclf_sbert_cuad_termination_clause", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])
 
df = spark.createDataFrame([["              ---------------------\n\n     This Agreement may be terminated immediately by Developer..."]]).toDF("clause_text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

## Results

```bash
+-------+
| result|
+-------+
|[termination]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_sbert_cuad_termination_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|22.8 MB|

## References

In-house annotations on CUAD dataset

## Benchmarking

```bash
label              precision    recall  f1-score   support
       other       1.00      1.00      1.00        41
 termination       1.00      1.00      1.00        40
    accuracy        -              -     1.00        81
   macro-avg       1.00      1.00      1.00        81
weighted-avg       1.00      1.00      1.00        81
```