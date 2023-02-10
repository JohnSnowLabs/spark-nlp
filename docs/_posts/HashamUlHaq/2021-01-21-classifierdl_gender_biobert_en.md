---
layout: model
title: Classifier for Genders - BIOBERT
author: John Snow Labs
name: classifierdl_gender_biobert
date: 2021-01-21
task: Text Classification
language: en
edition: Healthcare NLP 2.7.1
spark_version: 2.4
tags: [licensed, en, classifier, clinical]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model classifies the gender of the patient in the clinical document using context.

## Predicted Entities

`Female`, `Male`, `Unknown`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_GENDER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/21_Gender_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_gender_biobert_en_2.7.1_2.4_1611247084544.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/classifierdl_gender_biobert_en_2.7.1_2.4_1611247084544.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = Tokenizer()\
  .setInputCols(['document'])\
  .setOutputCol('token')

biobert_embeddings = BertEmbeddings().pretrained("biobert_pubmed_base_cased") \
  .setInputCols(["document", "token"])\
  .setOutputCol("bert_embeddings")

sentence_embeddings = SentenceEmbeddings() \
  .setInputCols(["document", "bert_embeddings"]) \
  .setOutputCol("sentence_bert_embeddings") \
  .setPoolingStrategy("AVERAGE")

genderClassifier = ClassifierDLModel.pretrained("classifierdl_gender_biobert", "en", "clinical/models") \
  .setInputCols(["document", "sentence_bert_embeddings"]) \
  .setOutputCol("gender")

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, biobert_embeddings, sentence_embeddings, genderClassifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

annotations = light_pipeline.fullAnnotate("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""")
```
```scala
val document_assembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

tokenizer = Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val biobert_embeddings = BertEmbeddings().pretrained("biobert_pubmed_base_cased")
  .setInputCols(Array("document","token"))
  .setOutputCol("bert_embeddings")

val sentence_embeddings = new SentenceEmbeddings()
  .setInputCols(Array("document", "bert_embeddings"))
  .setOutputCol("sentence_bert_embeddings")
  .setPoolingStrategy("AVERAGE")

val genderClassifier = ClassifierDLModel.pretrained("classifierdl_gender_biobert", "en", "clinical/models")
  .setInputCols(Array("document", "sentence_bert_embeddings"))
  .setOutputCol("gender")

val nlp_pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, biobert_embeddings, sentence_embeddings, genderClassifier))

val data = Seq("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.gender.biobert").predict("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""")
```

</div>

## Results

```bash
Female
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_gender_biobert|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|biobert_pubmed_base_cased|

## Data Source

This model is trained on more than four thousands clinical documents (radiology reports, pathology reports, clinical visits etc.), annotated internally.

## Benchmarking

```bash
label            precision    recall  f1-score   support
Female              0.9020    0.9364    0.9189       236
Male                0.8761    0.7857    0.8285       126
Unknown             0.7091    0.7647    0.7358        51
accuracy              -          -      0.8692       413
macro-avg           0.8291    0.8290    0.8277       413
weighted-avg        0.8703    0.8692    0.8687       413
```
