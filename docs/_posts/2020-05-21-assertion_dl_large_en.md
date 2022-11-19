---
layout: model
title: Detect Assertion Status (DL Large)
author: John Snow Labs
name: assertion_dl_large_en
date: 2020-05-21
task: Assertion Status
language: en
edition: Healthcare NLP 2.5.0
spark_version: 2.4
tags: [ner, en, clinical, licensed]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Deep learning named entity recognition model for assertions. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

## Predicted Entities
``hypothetical``, ``present``, ``absent``, ``possible``, ``conditional``, ``associated_with_someone_else``.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_large_en_2.5.0_2.4_1590022282256.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter, AssertionDLModel.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

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

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
    
clinical_assertion = AssertionDLModel.pretrained("assertion_dl_large", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, ner_converter, clinical_assertion])

model = nlpPipeline.fit(spark.createDataFrame([["Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer."]]).toDF("text"))

light_model = LightPipeline(model)

```

```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val nerConverter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("assertion_dl_large", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, word_embeddings, clinical_ner, nerConverter, clinical_assertion))

val data = Seq("""Patient with severe fever and sore throat. He shows no stomach pain and he maintained on an epidural and PCA for pain control. He also became short of breath with climbing a flight of stairs. After CT, lung tumor located at the right lower lobe. Father with Alzheimer.""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and an ``"assertion"`` column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select ``"ner_chunk.result"`` and ``"assertion.result"`` from your output dataframe.

```bash
           chunks  entities  assertion

0    severe fever   PROBLEM  present
1     sore throat   PROBLEM  present
2    stomach pain   PROBLEM  absent
3     an epidural TREATMENT  present
4             PCA TREATMENT  present
5    pain control   PROBLEM  present
6 short of breath   PROBLEM  conditional
7              CT      TEST  present
8      lung tumor   PROBLEM  present
9       Alzheimer   PROBLEM  associated_with_someone_else
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl_large|
|Type:|ner|
|Compatibility:|Spark NLP 2.5.0+|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, ner_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with 'embeddings_clinical'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.h2_title}
## Benchmarking
```bash

                              prec  rec   f1

                      absent  0.97  0.91  0.94
associated_with_someone_else  0.93  0.87  0.90
                 conditional  0.70  0.33  0.44
                hypothetical  0.91  0.82  0.86
                    possible  0.81  0.59  0.68
                     present  0.93  0.98  0.95

                   micro avg  0.93  0.93  0.93
                   macro avg  0.87  0.75  0.80
                   
```