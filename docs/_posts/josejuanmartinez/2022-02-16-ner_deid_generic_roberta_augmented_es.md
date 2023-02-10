---
layout: model
title: Detect PHI for Deidentification purposes (Spanish, reduced entities, augmented data, Roberta)
author: John Snow Labs
name: ner_deid_generic_roberta_augmented
date: 2022-02-16
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


Deidentification NER (Spanish) is a Named Entity Recognition model that annotates text to find protected health information that may need to be de-identified. It detects 8 entities (1 more than the `ner_deid_generic_roberta` ner model).


This NER model is trained with a combination of custom datasets, Spanish 2002 conLL, MeddoProf dataset, several data augmentation mechanisms and has been augmented with MEDDOCAN Spanish Deidentification corpus (compared to `ner_deid_generic_roberta` which does not include it). It's a generalized version of `ner_deid_subentity_roberta_augmented`.


This is a Roberta embeddings based model. You also have available the `ner_deid_generic_augmented` that uses Sciwi 300d embeddings.


## Predicted Entities


`CONTACT`, `NAME`, `DATE`, `ID`, `LOCATION`, `PROFESSION`, `AGE`, `SEX`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_roberta_augmented_es_3.3.4_3.0_1645006281743.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_roberta_augmented_es_3.3.4_3.0_1645006281743.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")


sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
.setInputCols(["document"])\
.setOutputCol("sentence")


tokenizer = nlp.Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")


roberta_embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")


clinical_ner = medical.NerModel.pretrained("ner_deid_generic_roberta_augmented", "es", "clinical/models")\
.setInputCols(["sentence","token","embeddings"])\
.setOutputCol("ner")


nlpPipeline = Pipeline(stages=[
documentAssembler,
sentenceDetector,
tokenizer,
roberta_embeddings,
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


roberta_embeddings = RoBertaEmbeddings.pretrained("roberta_base_biomedical", "es")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")


clinical_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented", "es", "clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")


val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, roberta_embeddings, clinical_ner))


val text = "Antonio Miguel Martínez, un varón de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos."


val df = Seq(text).toDF("text")


val results = pipeline.fit(df).transform(df)
```


{:.nlu-block}
```python
import nlu
nlu.load("es.med_ner.deid.generic.roberta").predict("""
Antonio Miguel Martínez, un varón de 35 años de edad, de profesión auxiliar de enfermería y nacido en Cadiz, España. Aún no estaba vacunado, se infectó con Covid-19 el dia 14 de Marzo y tuvo que ir al Hospital. Fue tratado con anticuerpos monoclonales en la Clinica San Carlos.
""")
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
|Model Name:|ner_deid_generic_roberta_augmented|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|16.3 MB|


## References


- Internal JSL annotated corpus
- [Spanish conLL](https://www.clips.uantwerpen.be/conll2002/ner/data/)
- [MeddoProf](https://temu.bsc.es/meddoprof/data/)
- [MeddoCan](https://temu.bsc.es/meddocan/)


## Benchmarking


```bash
label      tp     fp     fn   total  precision  recall       f1
CONTACT   177.0    3.0    6.0   183.0     0.9833  0.9672   0.9752
NAME  1963.0  159.0  123.0  2086.0     0.9251   0.941    0.933
DATE   953.0   18.0   16.0   969.0     0.9815  0.9835   0.9825
ORGANIZATION  2320.0  520.0  362.0  2682.0     0.8169   0.865   0.8403
ID    63.0    7.0    1.0    64.0        0.9  0.9844   0.9403
SEX   619.0   14.0    8.0   627.0     0.9779  0.9872   0.9825
LOCATION  2388.0  470.0  423.0  2811.0     0.8355  0.8495   0.8425
PROFESSION   233.0   15.0   28.0   261.0     0.9395  0.8927   0.9155
AGE   516.0   16.0    3.0   519.0     0.9699  0.9942   0.9819
macro       -      -      -       -          -       -  0.93263
micro       -      -      -       -          -       -  0.89427
```
