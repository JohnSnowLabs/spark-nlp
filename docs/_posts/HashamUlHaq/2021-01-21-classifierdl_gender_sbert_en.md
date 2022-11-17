---
layout: model
title: Classifier for Genders - SBERT
author: John Snow Labs
name: classifierdl_gender_sbert
date: 2021-01-21
task: Text Classification
language: en
edition: Healthcare NLP 2.7.1
spark_version: 2.4
tags: [en, licensed, classifier, clinical]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_gender_sbert_en_2.7.1_2.4_1611248306976.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli", 'en', 'clinical/models')\
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")

gender_classifier = ClassifierDLModel.pretrained( 'classifierdl_gender_sbert', 'en', 'clinical/models') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

nlp_pipeline = Pipeline(stages=[document_assembler, sbert_embedder, gender_classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""")
```



{:.nlu-block}
```python
import nlu
nlu.load("en.classify.gender.sbert").predict("""social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home. family history: shows a family history of breast cancer.""")
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
|Model Name:|classifierdl_gender_sbert|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|sbiobert_base_cased_mli|

## Data Source

This model is trained on more than four thousands clinical documents (radiology reports, pathology reports, clinical visits etc.), annotated internally.

## Benchmarking

```bash
precision    recall  f1-score   support

Female     0.9390    0.9747    0.9565       237
Male     0.9561    0.8720    0.9121       125
Unknown     0.8491    0.8824    0.8654        51

accuracy                         0.9322       413
macro avg     0.9147    0.9097    0.9113       413
weighted avg     0.9331    0.9322    0.9318       413
```