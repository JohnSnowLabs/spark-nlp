---
layout: model
title: Detect Assertion Status (assertion_dl_en)
author: John Snow Labs
name: assertion_dl_en
date: 2020-01-30
task: Assertion Status
language: en
edition: Healthcare NLP 2.4.0
spark_version: 2.4
tags: [clinical, licensed, ner, en]
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---
 
## Description

Deep learning named entity recognition model for assertions. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

{:.h2_title}
## Predicted Entities
``hypothetical``, ``present``, ``absent``, ``possible``, ``conditional``, ``associated_with_someone_else``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_en_2.4.0_2.4_1580237286004.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/assertion_dl_en_2.4.0_2.4_1580237286004.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel, NerConverter, AssertionDLModel.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentenceDetector = SentenceDetectorDLModel.pretrained()\
    .setInputCols("document")\
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

clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
    
nlpPipeline = Pipeline(stages=[document_assembler, 
                               sentenceDetector, 
                               tokenizer, 
                               word_embeddings, 
                               clinical_ner, 
                               ner_converter, 
                               clinical_assertion])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
light_result = LightPipeline(model).fullAnnotate('Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain')[0]

```

```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentenceDetector = SentenceDetectorDLModel.pretrained()
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

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")


val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                              sentenceDetector, 
                                              tokenizer, 
                                              word_embeddings, 
                                              clinical_ner, 
                                              ner_converter, 
                                              clinical_assertion))


val data = Seq("Patient has a headache for the last 2 weeks and appears anxious when she walks fast. No alopecia noted. She denies pain").toDS().toDF("text")
val result = pipeline.fit(data).transform(data)
```

</div>


## Results
The output is a dataframe with a sentence per row and an ``"assertion"`` column containing all of the assertion labels in the sentence. The assertion column also contains assertion character indices, and other metadata. To get only the entity chunks and assertion labels, without the metadata, select ``"ner_chunk.result"`` and ``"assertion.result"`` from your output dataframe.

```bash
|   | chunks     | entities | assertion   |
|---|------------|----------|-------------|
| 0 | a headache | PROBLEM  | present     |
| 1 | anxious    | PROBLEM  | conditional |
| 2 | alopecia   | PROBLEM  | absent      |
| 3 | pain       | PROBLEM  | absent      |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.0|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, ner_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|[en]|
|Case sensitive:|false|


## Data Source
Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with 'embeddings_clinical'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/


## Benchmarking
```bash
label                          prec  rec   f1   
absent                         0.94  0.87  0.91 
associated_with_someone_else   0.81  0.73  0.76 
conditional                    0.78  0.24  0.37 
hypothetical                   0.89  0.75  0.81 
possible                       0.70  0.52  0.60 
present                        0.91  0.97  0.94 
Macro-average                  0.84  0.68  0.73 
Micro-average                  0.91  0.91  0.91 
```
