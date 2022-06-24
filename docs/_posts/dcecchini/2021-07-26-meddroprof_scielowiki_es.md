---
layout: model
title: Professions & Occupations NER model in Spanish (meddroprof_scielowiki)
author: John Snow Labs
name: meddroprof_scielowiki
date: 2021-07-26
tags: [ner, licensed, prefessions, es, occupations]
task: Named Entity Recognition
language: es
edition: Spark NLP for Healthcare 3.1.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


NER model that detects professions and occupations in Spanish texts. Trained with the `embeddings_scielowiki_300d` embeddings, and the same `WordEmbeddingsModel` is needed in the pipeline.


## Predicted Entities


`ACTIVIDAD`, `PROFESION`, `SITUACION_LABORAL`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_PROFESSIONS_ES/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/meddroprof_scielowiki_es_3.1.3_3.0_1627328955264.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
        .setInputCol('text')\
        .setOutputCol('document')

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

embedings_stage = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d", "es", "clinical/models")\
        .setInputCols(["document","token"])\
        .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("meddroprof_scielowiki", "es", "clinical/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(['document', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence,
    tokenizer,
    embedings_stage,
    clinical_ner,
    ner_converter
])

example = spark.createDataFrame(pd.DataFrame({'text': ["""La paciente es la mayor de 2 hermanos, tiene un hermano de 13 años estudiando 1o ESO. Sus padres son ambos ATS , trabajan en diferentes centros de salud estudiando 1o ESO"""]}))

result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = new DocumentAssembler()
        .setInputCol('text')
        .setOutputCol('document')

val sentence = new SentenceDetector() 
    .setInputCols(["document"]) 
    .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols(["sentence"]) 
    .setOutputCol("token")

val embedings_stage = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d", "es", "clinical/models")
        .setInputCols(["document","token"])
        .setOutputCol("word_embeddings")

val clinical_ner = MedicalNerModel.pretrained("meddroprof_scielowiki", "es", "clinical/models")
        .setInputCols(["sentence", "token", "embeddings"])
        .setOutputCol("ner")

val ner_converter = new NerConverter() 
    .setInputCols(['document', 'token', 'ner']) 
    .setOutputCol('ner_chunk')

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence, tokenizer, embedings_stage, clinical_ner, ner_converter))

val data = Seq("""La paciente es la mayor de 2 hermanos, tiene un hermano de 13 años estudiando 1o ESO. Sus padres son ambos ATS , trabajan en diferentes centros de salud estudiando 1o ESO""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
+--------------------+----------+-------------------+
|                text|     token|         prediction|
+--------------------+----------+-------------------+
|La paciente es la...|        La|                  O|
|La paciente es la...|  paciente|                  O|
|La paciente es la...|        es|                  O|
|La paciente es la...|        la|                  O|
|La paciente es la...|     mayor|                  O|
|La paciente es la...|        de|                  O|
|La paciente es la...|         2|                  O|
|La paciente es la...|  hermanos|                  O|
|La paciente es la...|         ,|                  O|
|La paciente es la...|     tiene|                  O|
|La paciente es la...|        un|                  O|
|La paciente es la...|   hermano|                  O|
|La paciente es la...|        de|                  O|
|La paciente es la...|        13|                  O|
|La paciente es la...|      años|                  O|
|La paciente es la...|estudiando|B-SITUACION_LABORAL|
|La paciente es la...|        1o|I-SITUACION_LABORAL|
|La paciente es la...|       ESO|I-SITUACION_LABORAL|
|La paciente es la...|         .|                  O|
|La paciente es la...|       Sus|                  O|
|La paciente es la...|    padres|                  O|
|La paciente es la...|       son|                  O|
|La paciente es la...|     ambos|                  O|
|La paciente es la...|       ATS|        B-PROFESION|
|La paciente es la...|         ,|                  O|
|La paciente es la...|  trabajan|        B-PROFESION|
|La paciente es la...|        en|        I-PROFESION|
|La paciente es la...|diferentes|        I-PROFESION|
|La paciente es la...|   centros|        I-PROFESION|
|La paciente es la...|        de|        I-PROFESION|
|La paciente es la...|     salud|        I-PROFESION|
|La paciente es la...|estudiando|B-SITUACION_LABORAL|
|La paciente es la...|        1o|I-SITUACION_LABORAL|
|La paciente es la...|       ESO|I-SITUACION_LABORAL|
+--------------------+----------+-------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|meddroprof_scielowiki|
|Compatibility:|Spark NLP for Healthcare 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Dependencies:|embeddings_scielowiki_300d|


## Data Source


The model was trained with the [MEDDOPROF](https://temu.bsc.es/meddoprof/data/) data set:


> The MEDDOPROF corpus is a collection of 1844 clinical cases from over 20 different specialties annotated with professions and employment statuses. The corpus was annotated by a team composed of linguists and clinical experts following specially prepared annotation guidelines, after several cycles of quality control and annotation consistency analysis before annotating the entire dataset. Figure 1 shows a screenshot of a sample manual annotation generated using the brat annotation tool.


Reference:


```
@article{meddoprof,
    title={NLP applied to occupational health: MEDDOPROF shared task at IberLEF 2021 on automatic recognition, classification and normalization of professions and occupations from medical texts},
    author={Lima-López, Salvador and Farré-Maduell, Eulàlia and Miranda-Escalada, Antonio and Brivá-Iglesias, Vicent and Krallinger, Martin},
journal = {Procesamiento del Lenguaje Natural},
volume = {67},
    year={2021}
}
```


## Benchmarking


```bash
label               precision recall f1-score support
B-ACTIVIDAD         0.82      0.36   0.50     25     
B-PROFESION         0.87      0.75   0.81     634    
B-SITUACION_LABORAL 0.79      0.67   0.72     310    
I-ACTIVIDAD         0.86      0.43   0.57     58     
I-PROFESION         0.87      0.80   0.83     944    
I-SITUACION_LABORAL 0.74      0.71   0.73     407    
O                   1.00      1.00   1.00     139880 
accuracy            -         -      0.99     142258 
macro-avg           0.85      0.67   0.74     142258 
weighted-avg        0.99      0.99   0.99     142258 
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI4NDIwMTM0NF19
-->