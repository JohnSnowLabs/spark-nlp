---
layout: model
title: Detect PHI for Deidentification purposes (Spanish, reduced entities, Roberta Embeddings)
author: John Snow Labs
name: ner_deid_generic_roberta
date: 2022-01-17
tags: [deid, es, licensed]
task: De-identification
language: es
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. 


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 7 entities. This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf dataset and several data augmentation mechanisms, it's a reduced version of `ner_deid_subentity_roberta` and uses Roberta Clinical Embeddings.


## Predicted Entities


`CONTACT`, `NAME`, `DATE`, `ID`, `LOCATION`, `PROFESSION`, `AGE`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_roberta_es_3.3.4_3.0_1642437901644.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

# Feel free to experiment with multilingual or Spanish nlp.SentenceDetector instead
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

roberta_embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = medical.NerModel.pretrained("ner_deid_generic_roberta", "es", "clinical/models")\
        .setInputCols(["sentence","token","embeddings"])\
        .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        roberta_embeddings,
        clinical_ner])

text = ['''
Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
''']

df = spark.createDataFrame([text]).toDF("text")

results = nlpPipeline.fit(df).transform(df)
```
```scala
val documentAssembler = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")
        .setInputCols(Array("document"))
        .setOutputCol("sentence")

val tokenizer = new Tokenizer()
        .setInputCols(Array("sentence"))
        .setOutputCol("token")

val roberta_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_deid_generic_roberta", "es", "clinical/models")
        .setInputCols(Array("sentence","token","embeddings"))
        .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, roberta_embeddings, clinical_ner))

val text = """Antonio Pérez Juan, nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14/03/2020 y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."""

val df = Seq(text).toDS.toDF("text")

val results = pipeline.fit(df).transform(df)
```
</div>


## Results


```bash
+------------+----------+
|       token| ner_label|
+------------+----------+
|     Antonio|    B-NAME|
|       Pérez|    I-NAME|
|        Juan|    I-NAME|
|           ,|         O|
|      nacido|         O|
|          en|         O|
|       Cadiz|B-LOCATION|
|           ,|         O|
|      España|B-LOCATION|
|           .|         O|
|         Aún|         O|
|          no|         O|
|      estaba|         O|
|    vacunado|         O|
|           ,|         O|
|          se|         O|
|     infectó|         O|
|         con|         O|
|    Covid-19|         O|
|          el|         O|
|         dia|         O|
|  14/03/2020|    B-DATE|
|           y|         O|
|        tuvo|         O|
|         que|         O|
|          ir|         O|
|          al|         O|
|    Hospital|         O|
|         Fue|         O|
|     tratado|         O|
|         con|         O|
| anticuerpos|         O|
|monoclonales|         O|
|          en|         O|
|          la|         O|
|     Clinica|B-LOCATION|
|         San|I-LOCATION|
|      Carlos|I-LOCATION|
|           .|         O|
+------------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_generic_roberta|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|16.3 MB|
|Dependencies:|roberta_base_biomedical|


## Data Source


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)


## Benchmarking


```bash
     label      tp     fp     fn   total  precision  recall      f1
   CONTACT   171.0   10.0    3.0   174.0     0.9448  0.9828  0.9634
      NAME  2732.0  198.0  219.0  2951.0     0.9324  0.9258  0.9291
      DATE  1644.0   27.0   23.0  1667.0     0.9838  0.9862   0.985
        ID   114.0   11.0    7.0   121.0      0.912  0.9421  0.9268
  LOCATION  4850.0  623.0  594.0  5444.0     0.8862  0.8909  0.8885
PROFESSION   266.0   66.0  123.0   389.0     0.8012  0.6838  0.7379
       AGE   303.0   50.0   45.0   348.0     0.8584  0.8707  0.8645
     macro     -      -       -      -         -       -     0.8993
     micro     -      -       -      -         -       -     0.9094
```
