---
layout: model
title: Classifier for Adverse Drug Events
author: John Snow Labs
name: classifierdl_ade_biobert
date: 2021-01-21
task: Text Classification
language: en
edition: Healthcare NLP 2.7.1
spark_version: 2.4
tags: [licensed, clinical, en, classifier]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify text/sentence in two categories:

- `True` : The sentence is talking about a possible ADE

- `False` : The sentences doesn’t have any information about an ADE.

## Predicted Entities

`True`, `False`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_ade_biobert_en_2.7.1_2.4_1611243410222.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = BertEmbeddings.pretrained('biobert_pubmed_base_cased')\
    .setInputCols(["document", 'token'])\
    .setOutputCol("word_embeddings")

sentence_embeddings = SentenceEmbeddings() \
    .setInputCols(["document", "word_embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")

classifier = ClassifierDLModel.pretrained('classifierdl_ade_biobert', 'en', 'clinical/models')\
    .setInputCols(['document', 'token', 'sentence_embeddings'])\
    .setOutputCol('class')

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings, sentence_embeddings, classifier])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(["I feel a bit drowsy & have a little blurred vision after taking an insulin", "I feel great after taking tylenol"])
```

```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
	.setInputCols(Array("document", "token"))
	.setOutputCol("word_embeddings")

val sentence_embeddings = new SentenceEmbeddings()
	.setInputCols(Array("document", "word_embeddings"))
	.setOutputCol("sentence_embeddings")
	.setPoolingStrategy("AVERAGE")

val classifier = ClassifierDLModel.pretrained("classifierdl_ade_biobert", "en", "clinical/models")
	.setInputCols(Array("document", "token", "sentence_embeddings"))
	.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, sentence_embeddings, classifier))

val data = Seq("""I feel a bit drowsy & have a little blurred vision after taking an insulin, I feel great after taking tylenol""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.classify.ade.biobert").predict("""I feel a bit drowsy & have a little blurred vision after taking an insulin""")
```

</div>

## Results

```bash
|   | text                                                                       | label |
|--:|:---------------------------------------------------------------------------|:------|
| 0 | I feel a bit drowsy & have a little blurred vision after taking an insulin | True  |
| 1 | I feel great after taking tylenol                                          | False |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_ade_biobert|
|Compatibility:|Healthcare NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|biobert_pubmed_base_cased|

## Data Source

Trained on a custom dataset comprising of CADEC, DRUG-AE and Twimed.

## Benchmarking

```bash
label            precision    recall  f1-score   support
False       	  0.96        0.94      0.95      6923
True       	  0.71        0.79      0.75      1359
micro-avg         0.91        0.91      0.91      8282
macro-avg         0.83        0.86      0.85      8282
weighted-avg      0.92        0.91      0.91      8282
```
