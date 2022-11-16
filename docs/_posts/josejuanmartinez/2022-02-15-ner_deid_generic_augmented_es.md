---
layout: model
title: Detect PHI for Deidentification purposes (Spanish, reduced entities, augmented data)
author: John Snow Labs
name: ner_deid_generic_augmented
date: 2022-02-15
tags: [deid, es, licensed]
task: De-identification
language: es
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Named Entity Recognition annotators allow for a generic model to be trained by using a Deep Learning architecture (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN. 


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 8 entities (1 more than the `ner_deid_generic` ner model).


This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf dataset, several data augmentation mechanisms and has been augmented with MEDDOCAN Spanish Deidentification corpus (compared to `ner_deid_generic` which does not include it). It's a generalized version of `ner_deid_subentity_augmented`.


## Predicted Entities


`CONTACT`, `NAME`, `DATE`, `ID`, `LOCATION`, `PROFESSION`, `AGE`, `SEX`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_augmented_es_3.3.4_2.4_1644925864218.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")


# Feel free to experiment with multilingual or Spanish nlp.SentenceDetector instead
sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")


tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")


embeddings = nlp.WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("word_embeddings")


clinical_ner = medical.NerModel.pretrained("ner_deid_generic_augmented", "es", "clinical/models")\
        .setInputCols(["sentence","token","word_embeddings"])\
        .setOutputCol("ner")


nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        clinical_ner])


text = ['''
Antonio Miguel Martínez, un varón de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
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


embeddings = WordEmbeddingsModel.pretrained("embeddings_sciwiki_300d","es","clinical/models")
	.setInputCols(Array("sentence","token"))
	.setOutputCol("word_embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "es", "clinical/models")
        .setInputCols(Array("sentence","token","word_embeddings"))
        .setOutputCol("ner")


val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings, clinical_ner))


val text = "Antonio Miguel Martínez, un varón de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."


val df = Seq(text).toDF("text")


val results = pipeline.fit(df).transform(df)
```
</div>


## Results


```bash
+------------+------------+
|       token|   ner_label|
+------------+------------+
|     Antonio|      B-NAME|
|      Miguel|      I-NAME|
|    Martínez|      I-NAME|
|           ,|           O|
|          un|       B-SEX|
|       varón|       I-SEX|
|          de|           O|
|          35|       B-AGE|
|        años|           O|
|          de|           O|
|        edad|           O|
|           ,|           O|
|          de|           O|
|   profesión|           O|
|    auxiliar|B-PROFESSION|
|          de|I-PROFESSION|
|  enfermería|I-PROFESSION|
|           y|           O|
|      nacido|           O|
|          en|           O|
|       Cadiz|  B-LOCATION|
|           ,|           O|
|      España|  B-LOCATION|
|           .|           O|
|         Aún|           O|
|          no|           O|
|      estaba|           O|
|    vacunado|           O|
|           ,|           O|
|          se|           O|
|     infectó|           O|
|         con|           O|
|    Covid-19|B-PROFESSION|
|          el|           O|
|         dia|           O|
|          14|           O|
|          de|           O|
|       Marzo|           O|
|           y|           O|
|        tuvo|           O|
|         que|           O|
|          ir|           O|
|          al|           O|
|    Hospital|  B-LOCATION|
|         Fue|           O|
|     tratado|           O|
|         con|           O|
| anticuerpos|           O|
|monoclonales|           O|
|          en|           O|
|          la|           O|
|     Clinica|  B-LOCATION|
|         San|  I-LOCATION|
|      Carlos|  I-LOCATION|
|           .|           O|
+------------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_deid_generic_augmented|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|15.0 MB|


## References


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)
- [MeddoCan](https://temu.bsc.es/meddocan/)


## Benchmarking


```bash
       label      tp     fp     fn   total  precision  recall      f1
     CONTACT   185.0    3.0    0.0   185.0      0.984     1.0   0.992
        NAME  2066.0  138.0  106.0  2172.0     0.9374  0.9512  0.9442
        DATE  1017.0   18.0   18.0  1035.0     0.9826  0.9826  0.9826
ORGANIZATION  2468.0  482.0  332.0  2800.0     0.8366  0.8814  0.8584
          ID    65.0    5.0    3.0    68.0     0.9286  0.9559   0.942
         SEX   678.0    8.0   15.0   693.0     0.9883  0.9784  0.9833
    LOCATION  2532.0  358.0  420.0  2952.0     0.8761  0.8577  0.8668
  PROFESSION   246.0    9.0   31.0   277.0     0.9647  0.8881  0.9248
         AGE   547.0    8.0    9.0   556.0     0.9856  0.9838  0.9847
       macro       -      -      -       -          -       -  0.9421
       micro       -      -      -       -          -       -  0.9092
```
