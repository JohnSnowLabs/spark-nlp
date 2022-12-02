---
layout: model
title: Classifier for Adverse Drug Events
author: John Snow Labs
name: classifierdl_ade_biobert
date: 2020-09-30
task: Text Classification
language: en
edition: Healthcare NLP 2.6.2
spark_version: 2.4
tags: [classifier, en, clinical, licensed]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model classifies if a text is ADE-related (``True``) or not (``False``).

{:.h2_title}
## Predicted Entities
``True``, ``False``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/classifierdl_ade_biobert_en_2.6.0_2.4_1601594685053.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use
To classify your text if it is ADE-related, you can use this model as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, BertEmbeddings (``biobert_pubmed_base_cased``), SentenceEmbeddings, ClassifierDLModel.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
...
embeddings = BertEmbeddings.pretrained('biobert_pubmed_base_cased')\
    .setInputCols(["document", 'token'])\
    .setOutputCol("word_embeddings")

sentence_embeddings = SentenceEmbeddings() \
      .setInputCols(["document", "word_embeddings"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

classifier = ClassifierDLModel.pretrained('classifierdl_ade_biobert', 'en', 'clinical/models')\
    .setInputCols(['document', 'token', 'sentence_embeddings']).setOutputCol('class')

nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, embeddings, sentence_embeddings, classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate("I feel a bit drowsy & have a little blurred vision after taking an insulin")

```
```scala
...
val embeddings = BertEmbeddings.pretrained('biobert_pubmed_base_cased')
    .setInputCols(Array("document", 'token'))
    .setOutputCol("word_embeddings")
val sentence_embeddings = SentenceEmbeddings() 
      .setInputCols(Array("document", "word_embeddings")) 
      .setOutputCol("sentence_embeddings") 
      .setPoolingStrategy("AVERAGE")
val classsifierADE = ClassifierDLModel.pretrained("classifierdl_ade_biobert", "en", "clinical/models")
      .setInputCols(Array("sentence", "sentence_embeddings")) 
      .setOutputCol("class")
val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, sentence_embeddings, classifierADE))
val data = Seq("I feel a bit drowsy & have a little blurred vision after taking an insulin").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.ade.biobert").predict("""I feel a bit drowsy & have a little blurred vision after taking an insulin""")
```

</div>

{:.h2_title}
## Results
``True`` : The sentence is talking about a possible ADE

``False`` : The sentences doesn't have any information about an ADE.

```bash
'True'
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_ade_biobert|
|Type:|ClassifierDLModel|
|Compatibility:|Healthcare NLP 2.6.2 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|[en]|
|Case sensitive:|True|

{:.h2_title}
## Data Source
Trained on a custom dataset comprising of CADEC, DRUG-AE, Twimed using ``biobert_pubmed_base_cased`` embeddings.

{:.h2_title}
## Benchmarking
```bash
|    | label            | prec   | rec    | f1     |
|---:|-----------------:|-------:|-------:|-------:|
|  0 | False            | 0.9469 | 0.9327 | 0.9398 | 
|  1 | True             | 0.7603 | 0.8030 | 0.7811 | 
|  2 | Macro-average    | 0.8536 | 0.8679 | 0.8604 |
|  3 | Weighted-average | 0.9077 | 0.9055 | 0.9065 |
```