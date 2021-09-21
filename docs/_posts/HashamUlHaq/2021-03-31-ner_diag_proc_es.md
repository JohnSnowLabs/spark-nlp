---
layout: model
title: Detect Diagnoses and Procedures (Spanish)
author: John Snow Labs
name: ner_diag_proc
date: 2021-03-31
tags: [ner, clinical, licensed, es]
task: Named Entity Recognition
language: es
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.Pretrained named entity recognition deep learning model for diagnostics and procedures in spanish

## Predicted Entities

``DIAGNOSTICO``, ``PROCEDIMIENTO``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DIAG_PROC_ES/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DIAG_PROC_ES.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_diag_proc_es_3.0.0_3.0_1617208422892.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embed = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d","es","clinical/models")\
	.setInputCols(["document","token"])\
	.setOutputCol("word_embeddings")
model = MedicalNerModel.pretrained("ner_diag_proc","es","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embed, model, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([['HISTORIA DE ENFERMEDAD ACTUAL: El Sr. Smith es un hombre blanco veterano de 60 años con múltiples comorbilidades, que tiene antecedentes de cáncer de vejiga diagnosticado hace aproximadamente dos años por el Hospital VA. Allí se sometió a una resección. Debía ser ingresado en el Hospital de Día para una cistectomía. Fue visto en la Clínica de Urología y Clínica de Radiología el 02/04/2003. CURSO DE HOSPITAL: El Sr. Smith se presentó en el Hospital de Día antes de la cirugía de Urología. En evaluación, EKG, ecocardiograma fue anormal, se obtuvo una consulta de Cardiología. Luego se procedió a una resonancia magnética de estrés con adenosina cardíaca, la misma fue positiva para isquemia inducible, infarto subendocárdico inferolateral leve a moderado con isquemia peri-infarto. Además, se observa isquemia inducible en el tabique lateral inferior. El Sr. Smith se sometió a un cateterismo del corazón izquierdo, que reveló una enfermedad de las arterias coronarias de dos vasos. La RCA, proximal estaba estenosada en un 95% y la distal en un 80% estenosada. La LAD media estaba estenosada en un 85% y la LAD distal estaba estenosada en un 85%. Se colocaron cuatro stents de metal desnudo Multi-Link Vision para disminuir las cuatro lesiones al 0%. Después de la intervención, el Sr. Smith fue admitido en 7 Ardmore Tower bajo el Servicio de Cardiología bajo la dirección del Dr. Hart. El Sr. Smith tuvo un curso hospitalario post-intervención sin complicaciones. Se mantuvo estable para el alta hospitalaria el 07/02/2003 con instrucciones de tomar Plavix diariamente durante un mes y Urología está al tanto de lo mismo.']], ["text"]))
```
```scala
...
val embed = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d","es","clinical/models")
	.setInputCols(Array("document","token"))
	.setOutputCol("word_embeddings")
val model = MedicalNerModel.pretrained("ner_diag_proc","es","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embed, model, ner_converter))
val data = Seq("HISTORIA DE ENFERMEDAD ACTUAL: El Sr. Smith es un hombre blanco veterano de 60 años con múltiples comorbilidades, que tiene antecedentes de cáncer de vejiga diagnosticado hace aproximadamente dos años por el Hospital VA. Allí se sometió a una resección. Debía ser ingresado en el Hospital de Día para una cistectomía. Fue visto en la Clínica de Urología y Clínica de Radiología el 02/04/2003. CURSO DE HOSPITAL: El Sr. Smith se presentó en el Hospital de Día antes de la cirugía de Urología. En evaluación, EKG, ecocardiograma fue anormal, se obtuvo una consulta de Cardiología. Luego se procedió a una resonancia magnética de estrés con adenosina cardíaca, la misma fue positiva para isquemia inducible, infarto subendocárdico inferolateral leve a moderado con isquemia peri-infarto. Además, se observa isquemia inducible en el tabique lateral inferior. El Sr. Smith se sometió a un cateterismo del corazón izquierdo, que reveló una enfermedad de las arterias coronarias de dos vasos. La RCA, proximal estaba estenosada en un 95% y la distal en un 80% estenosada. La LAD media estaba estenosada en un 85% y la LAD distal estaba estenosada en un 85%. Se colocaron cuatro stents de metal desnudo Multi-Link Vision para disminuir las cuatro lesiones al 0%. Después de la intervención, el Sr. Smith fue admitido en 7 Ardmore Tower bajo el Servicio de Cardiología bajo la dirección del Dr. Hart. El Sr. Smith tuvo un curso hospitalario post-intervención sin complicaciones. Se mantuvo estable para el alta hospitalaria el 07/02/2003 con instrucciones de tomar Plavix diariamente durante un mes y Urología está al tanto de lo mismo.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------+-------------+
|chunk                 |ner_label    |
+----------------------+-------------+
|ENFERMEDAD            |DIAGNOSTICO  |
|cáncer de vejiga      |DIAGNOSTICO  |
|resección             |PROCEDIMIENTO|
|cistectomía           |PROCEDIMIENTO|
|estrés                |DIAGNOSTICO  |
|infarto subendocárdico|DIAGNOSTICO  |
|enfermedad            |DIAGNOSTICO  |
|arterias coronarias   |DIAGNOSTICO  |
+----------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_diag_proc|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|es|