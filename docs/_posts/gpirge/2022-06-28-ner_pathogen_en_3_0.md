---
layout: model
title: Detect Pathogen, Medical Condition and Medicine
author: John Snow Labs
name: ner_pathogen
date: 2022-06-28
tags: [licensed, clinical, en, ner, pathogen, medical_condition, medicine]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained named entity recognition (NER) model is a deep learning model for detecting medical conditions (influenza, headache, malaria, etc), medicine (aspirin, penicillin, methotrexate) and pathogens (Corona Virus, Zika Virus, E. Coli, etc) in clinical texts. It is trained by using `MedicalNerApproach` annotator that allows to train generic NER models based on Neural Networks.

## Predicted Entities

`Pathogen`, `MedicalCondition`, `Medicine`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_pathogen_en_4.0.0_3.0_1656419618392.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetectorDL = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl_healthcare", "en", 'clinical/models') \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical" ,"en", "clinical/models")\
    .setInputCols(["sentence","token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_pathogen", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[document_assembler,
        sentenceDetectorDL,
        tokenizer,
        word_embeddings,
        ner_model, 
        ner_converter])

data = spark.createDataFrame([["""Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical" ,"en", "clinical/models")
    .setInputCols(Array("sentence","token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_pathogen", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                            sentence_detector, 
                                            tokenizer, 
                                            word_embeddings, 
                                            ner_model, 
                                            ner_converter))

val data = Seq("""Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.pathogen").predict("""Racecadotril is an antisecretory medication and it has better tolerability than loperamide. Diarrhea is the condition of having loose, liquid or watery bowel movements each day. Signs of dehydration often begin with loss of the normal stretchiness of the skin.  This can progress to loss of skin color, a fast heart rate as it becomes more severe.  While it has been speculated that rabies virus, Lyssavirus and Ephemerovirus could be transmitted through aerosols, studies have concluded that this is only feasible in limited conditions.""")
```

</div>

## Results

```bash
+---------------+----------------+
|chunk          |ner_label       |
+---------------+----------------+
|Racecadotril   |Medicine        |
|loperamide     |Medicine        |
|Diarrhea       |MedicalCondition|
|dehydration    |MedicalCondition|
|skin color     |MedicalCondition|
|fast heart rate|MedicalCondition|
|rabies virus   |Pathogen        |
|Lyssavirus     |Pathogen        |
|Ephemerovirus  |Pathogen        |
+---------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_pathogen|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|14.6 MB|

## References

Trained on [dataset](https://www.kaggle.com/datasets/finalepoch/medical-ner) to get a model for Named Entity Recognition.

## Benchmarking

```bash
label               tp   fp   fn     total  precision  recall     f1
Pathogen            15.0  3.0  9.0   24.0     0.8333   0.625  0.7143
Medicine            15.0  2.0  0.0   15.0     0.8824   1.0    0.9375
MedicalCondition    53.0  2.0  6.0   59.0     0.9636   0.8983 0.9298
macro               -     -    -     -        -        -      0.8605
micro               -     -    -     -        -        -      0.8782
```