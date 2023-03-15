---
layout: model
title: Pipeline to Detect Neoplasms
author: John Snow Labs
name: ner_neoplasms_pipeline
date: 2023-03-15
tags: [ner, clinical, licensed, es]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_neoplasms](https://nlp.johnsnowlabs.com/2021/03/31/ner_neoplasms_es.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_neoplasms_pipeline_es_4.3.0_3.2_1678879327865.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_neoplasms_pipeline_es_4.3.0_3.2_1678879327865.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_neoplasms_pipeline", "es", "clinical/models")

text = '''HISTORIA DE ENFERMEDAD ACTUAL: El Sr. Smith es un hombre blanco veterano de 60 años con múltiples comorbilidades, que tiene antecedentes de cáncer de vejiga diagnosticado hace aproximadamente dos años por el Hospital VA. Allí se sometió a una resección. Debía ser ingresado en el Hospital de Día para una cistectomía. Fue visto en la Clínica de Urología y Clínica de Radiología el 02/04/2003. CURSO DE HOSPITAL: El Sr. Smith se presentó en el Hospital de Día antes de la cirugía de Urología. En evaluación, EKG, ecocardiograma fue anormal, se obtuvo una consulta de Cardiología. Luego se procedió a una resonancia magnética de estrés con adenosina cardíaca, la misma fue positiva para isquemia inducible, infarto subendocárdico inferolateral leve a moderado con isquemia peri-infarto. Además, se observa isquemia inducible en el tabique lateral inferior. El Sr. Smith se sometió a un cateterismo del corazón izquierdo, que reveló una enfermedad de las arterias coronarias de dos vasos. La RCA, proximal estaba estenosada en un 95% y la distal en un 80% estenosada. La LAD media estaba estenosada en un 85% y la LAD distal estaba estenosada en un 85%. Se colocaron cuatro stents de metal desnudo Multi-Link Vision para disminuir las cuatro lesiones al 0%. Después de la intervención, el Sr. Smith fue admitido en 7 Ardmore Tower bajo el Servicio de Cardiología bajo la dirección del Dr. Hart. El Sr. Smith tuvo un curso hospitalario post-intervención sin complicaciones. Se mantuvo estable para el alta hospitalaria el 07/02/2003 con instrucciones de tomar Plavix diariamente durante un mes y Urología está al tanto de lo mismo.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_neoplasms_pipeline", "es", "clinical/models")

val text = "HISTORIA DE ENFERMEDAD ACTUAL: El Sr. Smith es un hombre blanco veterano de 60 años con múltiples comorbilidades, que tiene antecedentes de cáncer de vejiga diagnosticado hace aproximadamente dos años por el Hospital VA. Allí se sometió a una resección. Debía ser ingresado en el Hospital de Día para una cistectomía. Fue visto en la Clínica de Urología y Clínica de Radiología el 02/04/2003. CURSO DE HOSPITAL: El Sr. Smith se presentó en el Hospital de Día antes de la cirugía de Urología. En evaluación, EKG, ecocardiograma fue anormal, se obtuvo una consulta de Cardiología. Luego se procedió a una resonancia magnética de estrés con adenosina cardíaca, la misma fue positiva para isquemia inducible, infarto subendocárdico inferolateral leve a moderado con isquemia peri-infarto. Además, se observa isquemia inducible en el tabique lateral inferior. El Sr. Smith se sometió a un cateterismo del corazón izquierdo, que reveló una enfermedad de las arterias coronarias de dos vasos. La RCA, proximal estaba estenosada en un 95% y la distal en un 80% estenosada. La LAD media estaba estenosada en un 85% y la LAD distal estaba estenosada en un 85%. Se colocaron cuatro stents de metal desnudo Multi-Link Vision para disminuir las cuatro lesiones al 0%. Después de la intervención, el Sr. Smith fue admitido en 7 Ardmore Tower bajo el Servicio de Cardiología bajo la dirección del Dr. Hart. El Sr. Smith tuvo un curso hospitalario post-intervención sin complicaciones. Se mantuvo estable para el alta hospitalaria el 07/02/2003 con instrucciones de tomar Plavix diariamente durante un mes y Urología está al tanto de lo mismo."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk   |   begin |   end | ner_label            |   confidence |
|---:|:------------|--------:|------:|:---------------------|-------------:|
|  0 | cáncer      |     140 |   145 | MORFOLOGIA_NEOPLASIA |       0.9997 |
|  1 | Multi-Link  |    1195 |  1204 | MORFOLOGIA_NEOPLASIA |       0.574  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_neoplasms_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|383.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter