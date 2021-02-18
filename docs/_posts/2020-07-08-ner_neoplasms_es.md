---
layout: model
title: Detect Neoplasms 
author: John Snow Labs
name: ner_neoplasms
class: NerDLModel
language: es
repository: clinical/models
date: 2020-07-08
task: Named Entity Recognition
edition: Spark NLP for Healthcare 2.5.3
tags: [clinical,licensed,ner,es]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.
Neoplasms NER is a Named Entity Recognition model that annotates text to find references to tumors. The only entity it annotates is MalignantNeoplasm. Neoplasms NER is trained with the 'embeddings_scielowiki_300d' word embeddings model, so be sure to use the same embeddings in the pipeline.

## Predicted Entities 
``MORFOLOGIA_NEOPLASIA``

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_neoplasms_es_2.5.3_2.4_1594168624415.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
embed = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d","es","clinical/models")\
	.setInputCols(["document","token"])\
	.setOutputCol("word_embeddings")
model = NerDLModel.pretrained("ner_neoplasms","es","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embed, model, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""HISTORIA DE ENFERMEDAD ACTUAL: El Sr. Smith es un hombre blanco veterano de 60 años con múltiples comorbilidades, que tiene antecedentes de cáncer de vejiga diagnosticado hace aproximadamente dos años por el Hospital VA. Allí se sometió a una resección. Debía ser ingresado en el Hospital de Día para una cistectomía. Fue visto en la Clínica de Urología y Clínica de Radiología el 02/04/2003. CURSO DE HOSPITAL: El Sr. Smith se presentó en el Hospital de Día antes de la cirugía de Urología. En evaluación, EKG, ecocardiograma fue anormal, se obtuvo una consulta de Cardiología. Luego se procedió a una resonancia magnética de estrés con adenosina cardíaca, la misma fue positiva para isquemia inducible, infarto subendocárdico inferolateral leve a moderado con isquemia peri-infarto. Además, se observa isquemia inducible en el tabique lateral inferior. El Sr. Smith se sometió a un cateterismo del corazón izquierdo, que reveló una enfermedad de las arterias coronarias de dos vasos. La RCA, proximal estaba estenosada en un 95% y la distal en un 80% estenosada. La LAD media estaba estenosada en un 85% y la LAD distal estaba estenosada en un 85%. Se colocaron cuatro stents de metal desnudo Multi-Link Vision para disminuir las cuatro lesiones al 0%. Después de la intervención, el Sr. Smith fue admitido en 7 Ardmore Tower bajo el Servicio de Cardiología bajo la dirección del Dr. Hart. El Sr. Smith tuvo un curso hospitalario post-intervención sin complicaciones. Se mantuvo estable para el alta hospitalaria el 07/02/2003 con instrucciones de tomar Plavix diariamente durante un mes y Urología está al tanto de lo mismo."""]})))
```

```scala
...
val embed = WordEmbeddingsModel.pretrained("embeddings_scielowiki_300d","es","clinical/models")
	.setInputCols(Array("document","token"))
	.setOutputCol("word_embeddings")
val model = NerDLModel.pretrained("ner_neoplasms","es","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embed, model, ner_converter))
val result = pipeline.fit(Seq.empty["HISTORIA DE ENFERMEDAD ACTUAL: El Sr. Smith es un hombre blanco veterano de 60 años con múltiples comorbilidades, que tiene antecedentes de cáncer de vejiga diagnosticado hace aproximadamente dos años por el Hospital VA. Allí se sometió a una resección. Debía ser ingresado en el Hospital de Día para una cistectomía. Fue visto en la Clínica de Urología y Clínica de Radiología el 02/04/2003. CURSO DE HOSPITAL: El Sr. Smith se presentó en el Hospital de Día antes de la cirugía de Urología. En evaluación, EKG, ecocardiograma fue anormal, se obtuvo una consulta de Cardiología. Luego se procedió a una resonancia magnética de estrés con adenosina cardíaca, la misma fue positiva para isquemia inducible, infarto subendocárdico inferolateral leve a moderado con isquemia peri-infarto. Además, se observa isquemia inducible en el tabique lateral inferior. El Sr. Smith se sometió a un cateterismo del corazón izquierdo, que reveló una enfermedad de las arterias coronarias de dos vasos. La RCA, proximal estaba estenosada en un 95% y la distal en un 80% estenosada. La LAD media estaba estenosada en un 85% y la LAD distal estaba estenosada en un 85%. Se colocaron cuatro stents de metal desnudo Multi-Link Vision para disminuir las cuatro lesiones al 0%. Después de la intervención, el Sr. Smith fue admitido en 7 Ardmore Tower bajo el Servicio de Cardiología bajo la dirección del Dr. Hart. El Sr. Smith tuvo un curso hospitalario post-intervención sin complicaciones. Se mantuvo estable para el alta hospitalaria el 07/02/2003 con instrucciones de tomar Plavix diariamente durante un mes y Urología está al tanto de lo mismo."].toDS.toDF("text")).transform(data)
```
</div>

## Results

```bash
+------+--------------------+
|chunk |ner_label           |
+------+--------------------+
|cáncer|MORFOLOGIA_NEOPLASIA|
+------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | ner_neoplasms                    |
| Type:    | NerDLModel                       |
| Compatibility:  | Spark NLP 2.5.3+                            |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [sentence, token, word_embeddings] |
|Output labels:        | [ner]                              |
| Language:       | es                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_scielowiki_300d       |

{:.h2_title}
## Data Source
Named Entity Recognition model for Neoplasic Morphology
https://temu.bsc.es/cantemist/