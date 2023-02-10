---
layout: model
title: Legal Force Majeure Clause Binary Classifier (CUAD dataset)
author: John Snow Labs
name: legclf_cuad_force_majeure_clause
date: 2022-11-30
tags: [en, licensed]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Binary Classifier (True, False) for the `force-majeure` clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Spark NLP for Legal Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Legal/1.Tokenization_Splitting.ipynb)), namely:
- Paragraph splitting (by multiline);
- Splitting by headers / subheaders;
- etc.

Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

There are other models in this dataset with similar title, but the difference is the dataset it was trained on. This one was trained with `cuad` dataset.

## Predicted Entities

`other`, `force-majeure`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/CLASSIFY_LEGAL_CLAUSES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_cuad_force_majeure_clause_en_1.0.0_3.0_1669806586316.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_cuad_force_majeure_clause_en_1.0.0_3.0_1669806586316.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

docClassifier = legal.ClassifierDLModel.pretrained("legclf_cuad_force_majeure_clause", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])
 
df = spark.createDataFrame([["10 . FORCE-MAJEURE 10.1 Except for the obligations to make any payment , required by this Contract ( which shall not be subject to relief under this item ), a Party shall not be in breach of this Contract and liable to the other Party for any failure to fulfil any obligation under this Contract to the extent any fulfillment has been interfered with , hindered , delayed , or prevented by any circumstance whatsoever , which is not reasonably within the control of and is unforeseeable by such Party and if such Party exercised due diligence , including acts of God , fire , flood , freezing , landslides , lightning , earthquakes , fire , storm , floods , washouts , and other natural disasters , wars ( declared or undeclared ), insurrections , riots , civil disturbances , epidemics , quarantine restrictions , blockade , embargo , strike , lockouts , labor disputes , or restrictions imposed by any government ."]]).toDF("clause_text")

model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

## Results

```bash
+-------+
| result|
+-------+
|[force-majeure]|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_cuad_force_majeure_clause|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[category]|
|Language:|en|
|Size:|23.3 MB|

## References

In-house annotations on Cuad dataset.

## Benchmarking

```bash
label               precision    recall  f1-score   support
force-majeure       0.97      0.94      0.95        31
        other       0.96      0.98      0.97        56
     accuracy      -           -          0.97        87
    macro-avg       0.97      0.96      0.96        87
 weighted-avg       0.97      0.97      0.97        87

```
