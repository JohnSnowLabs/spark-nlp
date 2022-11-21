---
layout: model
title: Detect Assertion Status (assertion_dl_healthcare) - supports confidence scores
author: John Snow Labs
name: assertion_dl_healthcare
date: 2021-01-26
task: Assertion Status
language: en
edition: Healthcare NLP 2.7.2
spark_version: 2.4
tags: [assertion, en, licensed, clinical, healthcare]
supported: true
annotator: AssertionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Assign assertion status to clinical entities extracted by NER based on their context in the text.

## Predicted Entities

`absent`, `present`, `conditional`, `associated_with_someone_else`, `hypothetical`, `possible`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_healthcare_en_2.7.2_2.4_1611646187271.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_healthcare", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

clinical_assertion = AssertionDLModel.pretrained("assertion_dl_healthcare", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

nlpPipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    clinical_ner,
    ner_converter,
    clinical_assertion
    ])

data = spark.createDataFrame([["""Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer."""]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols("document") 
    .setOutputCol("sentence") 

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_healthcare", "en", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence","token","ner"))
    .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("assertion_dl_healthcare","en", "clinical/models") 
    .setInputCols(Array("sentence", "ner_chunk", "embeddings")) 
    .setOutputCol("assertion")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, 
                                               sentenceDetector, 
                                               tokenizer, 
                                               word_embeddings, 
                                               clinical_ner, 
                                               ner_converter, 
                                               clinical_assertion))

val data = Seq("Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.assert.healthcare").predict("""Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer.""")
```

</div>

## Results

```bash
+---------------+---------+----------------------------+
|chunk          |ner_label|assertion                   |
+---------------+---------+----------------------------+
|severe fever   |PROBLEM  |present                     |
|sore throat    |PROBLEM  |present                     |
|stomach pain   |PROBLEM  |absent                      |
|an epidural    |TREATMENT|present                     |
|PCA            |TREATMENT|present                     |
|pain control   |TREATMENT|present                     |
|short of breath|PROBLEM  |conditional                 |
|CT             |TEST     |present                     |
|lung tumor     |PROBLEM  |present                     |
|Alzheimer      |PROBLEM  |associated_with_someone_else|
+---------------+---------+----------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl_healthcare|
|Compatibility:|Spark NLP 2.7.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|

## Data Source

Trained on i2b2 assertion data

## Benchmarking

```bash
label                            tp    fp    fn      prec       rec        f1
absent                          726    86    98  0.894089  0.881068  0.887531
present                        2544   232   119  0.916427  0.955314  0.935466
conditional                      18    13    37  0.580645  0.327273  0.418605
associated_with_someone_else     40     5     9  0.888889  0.816327  0.851064
hypothetical                    132    13    26  0.910345  0.835443  0.871287
possible                         96    45   105  0.680851  0.477612  0.561404
Macro-average                  3556   394   394  0.811874  0.715506  0.76065 
Micro-average                  3556   394   394  0.900253  0.900253  0.900253
```