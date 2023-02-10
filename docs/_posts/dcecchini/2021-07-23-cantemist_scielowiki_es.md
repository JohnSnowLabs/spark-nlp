---
layout: model
title: Detect tumor morphology in Spanish texts
author: John Snow Labs
name: cantemist_scielowiki
date: 2021-07-23
tags: [ner, licensed, oncology, es]
task: Named Entity Recognition
language: es
edition: Spark NLP for Healthcare 3.1.2
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description

Detect tumor morphology entities in Spanish text.


## Predicted Entities

`MORFOLOGIA_NEOPLASIA`, `O`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TUMOR_ES/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/cantemist_scielowiki_es_3.1.2_3.0_1627080305994.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/cantemist_scielowiki_es_3.1.2_3.0_1627080305994.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

clinical_ner = MedicalNerModel.pretrained("cantemist_scielowiki", "es", "clinical/models")\
    .setInputCols(["sentence", "token", "word_embeddings"])\
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

data = spark.createDataFrame([["""Anamnesis Paciente de 37 años de edad sin antecedentes patológicos ni quirúrgicos de interés. En diciembre de 2012 consultó al Servicio de Urgencias por un cuadro de cefalea aguda e hipostesia del hemicuerpo izquierdo de 15 días de evolución refractario a tratamiento. Exploración neurológica sin focalidad; fondo de ojo: papiledema unilateral. Se solicitaron una TC del SNC, que objetiva una LOE frontal derecha con afectación aparente del cuerpo calloso, y una RM del SNC, que muestra un extenso proceso expansivo intraparenquimatoso frontal derecho que infiltra la rodilla del cuerpo calloso, mal delimitada y sin componente necrótico. Tras la administración de contraste se apreciaban diferentes realces parcheados en la lesión, pero sin definirse una cápsula con aumento del flujo sanguíneo en la lesión, características compatibles con linfoma o astrocitoma anaplásico . El 3 de enero de 2013 se efectúa biopsia intraoperatoria, con diagnóstico histológico de astrocitoma anaplásico GIII"""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence = new SentenceDetector() 
    .setInputCols("document") 
    .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols("sentence") 
    .setOutputCol("token")

val embedings_stage = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d", "es", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("word_embeddings")

val clinical_ner = MedicalNerModel.pretrained("cantemist_scielowiki", "es", "clinical/models")
    .setInputCols(Array("sentence", "token", "word_embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter() 
    .setInputCols(Array("document", "token", "ner")) 
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence, tokenizer, embedings_stage, clinical_ner, ner_converter))

val data = Seq("""Anamnesis Paciente de 37 años de edad sin antecedentes patológicos ni quirúrgicos de interés. En diciembre de 2012 consultó al Servicio de Urgencias por un cuadro de cefalea aguda e hipostesia del hemicuerpo izquierdo de 15 días de evolución refractario a tratamiento. Exploración neurológica sin focalidad; fondo de ojo: papiledema unilateral. Se solicitaron una TC del SNC, que objetiva una LOE frontal derecha con afectación aparente del cuerpo calloso, y una RM del SNC, que muestra un extenso proceso expansivo intraparenquimatoso frontal derecho que infiltra la rodilla del cuerpo calloso, mal delimitada y sin componente necrótico. Tras la administración de contraste se apreciaban diferentes realces parcheados en la lesión, pero sin definirse una cápsula con aumento del flujo sanguíneo en la lesión, características compatibles con linfoma o astrocitoma anaplásico . El 3 de enero de 2013 se efectúa biopsia intraoperatoria, con diagnóstico histológico de astrocitoma anaplásico GIII""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
+---------------------+----------------------+
|              token  |          prediction  |
+---------------------+----------------------+
|          Anamnesis  |                   O  |
|           Paciente  |                   O  |
|                 de  |                   O  |
|                 37  |                   O  |
|               años  |                   O  |
|                 de  |                   O  |
|               edad  |                   O  |
|                sin  |                   O  |
|       antecedentes  |                   O  |
|        patológicos  |                   O  |
|                 ni  |                   O  |
|        quirúrgicos  |                   O  |
|                 de  |                   O  |
|            interés  |                   O  |
|                  .  |                   O  |
|                 En  |                   O  |
|          diciembre  |                   O  |
|                 de  |                   O  |
|               2012  |                   O  |
|           consultó  |                   O  |
|                 al  |                   O  |
|           Servicio  |                   O  |
|                 de  |                   O  |
|          Urgencias  |                   O  |
|                por  |                   O  |
|                 un  |                   O  |
|             cuadro  |                   O  |
|                 de  |                   O  |
|            cefalea  |                   O  |
|              aguda  |                   O  |
|                  e  |                   O  |
|         hipostesia  |                   O  |
|                del  |                   O  |
|         hemicuerpo  |                   O  |
|          izquierdo  |                   O  |
|                 de  |                   O  |
|                 15  |                   O  |
|               días  |                   O  |
|                 de  |                   O  |
|          evolución  |                   O  |
|        refractario  |                   O  |
|                  a  |                   O  |
|        tratamiento  |                   O  |
|                  .  |                   O  |
|        Exploración  |                   O  |
|        neurológica  |                   O  |
|                sin  |                   O  |
|          focalidad  |                   O  |
|                  ;  |                   O  |
|              fondo  |                   O  |
|                 de  |                   O  |
|                ojo  |                   O  |
|                  :  |                   O  |
|         papiledema  |                   O  |
|         unilateral  |                   O  |
|                  .  |                   O  |
|                 Se  |                   O  |
|        solicitaron  |                   O  |
|                una  |                   O  |
|                 TC  |                   O  |
|                del  |                   O  |
|                SNC  | B-MORFOLOGIA_NEOP... |
|                  ,  |                   O  |
|                que  |                   O  |
|           objetiva  |                   O  |
|                una  |                   O  |
|                LOE  |                   O  |
|            frontal  |                   O  |
|            derecha  |                   O  |
|                con  |                   O  |
|         afectación  | B-MORFOLOGIA_NEOP... |
|           aparente  | I-MORFOLOGIA_NEOP... |
|                del  | I-MORFOLOGIA_NEOP... |
|             cuerpo  | I-MORFOLOGIA_NEOP... |
|            calloso  | I-MORFOLOGIA_NEOP... |
|                  ,  |                   O  |
|                  y  |                   O  |
|                una  |                   O  |
|                 RM  | B-MORFOLOGIA_NEOP... |
|                del  | I-MORFOLOGIA_NEOP... |
|                SNC  | I-MORFOLOGIA_NEOP... |
|                  ,  |                   O  |
|                que  |                   O  |
|            muestra  |                   O  |
|                 un  |                   O  |
|            extenso  |                   O  |
|            proceso  | B-MORFOLOGIA_NEOP... |
|          expansivo  | I-MORFOLOGIA_NEOP... |
| intraparenquimatoso | I-MORFOLOGIA_NEOP... |
|            frontal  | I-MORFOLOGIA_NEOP... |
|            derecho  | I-MORFOLOGIA_NEOP... |
|                que  | I-MORFOLOGIA_NEOP... |
|           infiltra  | I-MORFOLOGIA_NEOP... |
|                 la  | I-MORFOLOGIA_NEOP... |
|            rodilla  | I-MORFOLOGIA_NEOP... |
|                del  | I-MORFOLOGIA_NEOP... |
|             cuerpo  | I-MORFOLOGIA_NEOP... |
|            calloso  | I-MORFOLOGIA_NEOP... |
|                  ,  |                   O  |
|                mal  |                   O  |
|         delimitada  |                   O  |
|                  y  |                   O  |
|                sin  |                   O  |
|         componente  |                   O  |
|          necrótico  |                   O  |
|                  .  |                   O  |
|               Tras  |                   O  |
|                 la  |                   O  |
|     administración  |                   O  |
|                 de  |                   O  |
|          contraste  |                   O  |
|                 se  |                   O  |
|         apreciaban  |                   O  |
|         diferentes  |                   O  |
|            realces  |                   O  |
|         parcheados  |                   O  |
|                 en  |                   O  |
|                 la  |                   O  |
|             lesión  |                   O  |
|                  ,  |                   O  |
|               pero  |                   O  |
|                sin  |                   O  |
|          definirse  |                   O  |
|                una  |                   O  |
|            cápsula  |                   O  |
|                con  |                   O  |
|            aumento  |                   O  |
|                del  |                   O  |
|              flujo  |                   O  |
|          sanguíneo  |                   O  |
|                 en  |                   O  |
|                 la  |                   O  |
|             lesión  |                   O  |
|                  ,  |                   O  |
|    características  |                   O  |
|        compatibles  |                   O  |
|                con  |                   O  |
|            linfoma  |                   O  |
|                  o  |                   O  |
|        astrocitoma  | B-MORFOLOGIA_NEOP... |
|         anaplásico  | I-MORFOLOGIA_NEOP... |
|                  .  |                   O  |
|                 El  |                   O  |
|                  3  |                   O  |
|                 de  |                   O  |
|              enero  |                   O  |
|                 de  |                   O  |
|               2013  |                   O  |
|                 se  |                   O  |
|            efectúa  |                   O  |
|            biopsia  |                   O  |
|    intraoperatoria  |                   O  |
|                  ,  |                   O  |
|                con  |                   O  |
|        diagnóstico  |                   O  |
|        histológico  |                   O  |
|                 de  |                   O  |
|        astrocitoma  | B-MORFOLOGIA_NEOP... |
|         anaplásico  | I-MORFOLOGIA_NEOP... |
|               GIII  | I-MORFOLOGIA_NEOP... |
+---------------------+----------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|cantemist_scielowiki|
|Compatibility:|Spark NLP for Healthcare 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, word_embeddings]|
|Output Labels:|[ner]|
|Language:|es|
|Dependencies:|embeddings_scielowiki_300d|


## Data Source


The model was trained with the [CANTEMIST](https://temu.bsc.es/cantemist/) data set:


> CANTEMIST is an annotated data set for oncology analysis in the Spanish language containing 1301 oncological clinical case reports with a total of 63,016 sentences and 1093,501 tokens. All documents of the corpus have been manually annotated by clinical experts with
mentions of tumor morphology (in Spanish, “morfología de neoplasia”). There are 16,030 tumor morphology mentions mapped to an eCIE-O code (850 unique codes)




References:




1. P. Ruas, A. Neves, V. D. Andrade, F. M. Couto, Lasigebiotm at cantemist: Named entity recognition and normalization of tumour morphology entities and clinical coding of Spanish health-related documents, in: Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2020), CEUR Workshop Proceedings, 2020


2. Antonio Miranda-Escalada, Eulàlia Farré-Maduell, Martin Krallinger. Named Entity Recognition, Concept Normalization and Clinical Coding: Overview of the Cantemist Track for Cancer Text Mining in Spanish, Corpus, Guidelines, Methods and Results. Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2020), CEUR Workshop Proceedings. 303-323 (2020).


## Benchmarking


```bash
label                  precision recall f1-score support
B-MORFOLOGIA_NEOPLASIA 0.94      0.73   0.83     2474   
I-MORFOLOGIA_NEOPLASIA 0.81      0.74   0.77     3169   
O                      0.99      1.00   1.00     283006 
accuracy               -         -      0.99     288649 
macro-avg              0.92      0.82   0.87     288649 
weighted-avg           0.99      0.99   0.99     288649 
```