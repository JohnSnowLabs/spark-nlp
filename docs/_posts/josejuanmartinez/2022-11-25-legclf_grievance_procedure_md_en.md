---
layout: model
title: Legal Grievance procedure Clause Binary Classifier (md)
author: John Snow Labs
name: legclf_grievance_procedure_md
date: 2022-11-25
tags: [en, legal, classification, document, agreement, contract, licensed]
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

This model is a Binary Classifier (True, False) for the `grievance-procedure` clause type. To use this model, make sure you provide enough context as an input. Adding Sentence Splitters to the pipeline will make the model see only sentences, not the whole text, so it's better to skip it, unless you want to do Binary Classification as sentence level.

If you have big legal documents, and you want to look for clauses, we recommend you to split the documents using any of the techniques available in our Spark NLP for Legal Workshop Tokenization & Splitting Tutorial (link [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Legal/1.Tokenization_Splitting.ipynb)), namely:
- Paragraph splitting (by multiline);
- Splitting by headers / subheaders;
- etc.

Take into consideration the embeddings of this model allows up to 512 tokens. If you have more than that, consider splitting in smaller pieces (you can also check the same tutorial link provided above).

This model can be combined with any of the other 200+ Legal Clauses Classifiers you will find in Models Hub, getting as an output a series of True/False values for each of the legal clause model you have added.

## Predicted Entities

`other`, `grievance-procedure`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/CLASSIFY_LEGAL_DOCUMENTS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_grievance_procedure_md_en_1.0.0_3.0_1669376484167.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_grievance_procedure_md_en_1.0.0_3.0_1669376484167.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

docClassifier = legal.ClassifierDLModel.pretrained("legclf_grievance_procedure_md", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("category")
    
nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    embeddings,
    docClassifier])
 
df = spark.createDataFrame([["YOUR TEXT HERE"]]).toDF("clause_text")
model = nlpPipeline.fit(df)
result = model.transform(df)
```

</div>

## Results

```bash
+-------+
| result|
+-------+
|[grievance-procedure]|
|[other]|
|[other]|
|[grievance-procedure]|

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_grievance_procedure_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.6 MB|

## References

Legal documents, scrapped from the Internet, and classified in-house + SEC documents + Lawinsider categorization

## Benchmarking

```bash
                     precision    recall  f1-score   support

grievance-procedure       1.00      0.96      0.98        24
              other       0.97      1.00      0.99        39

           accuracy                           0.98        63
          macro avg       0.99      0.98      0.98        63
       weighted avg       0.98      0.98      0.98        63

```
