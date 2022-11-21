---
layout: model
title: Clinical Deidentification (Spanish, augmented)
author: John Snow Labs
name: clinical_deidentification_augmented
date: 2022-02-16
tags: [deid, es, licensed]
task: De-identification
language: es
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline is trained with sciwiki_300d embeddings and can be used to deidentify PHI information from medical texts in Spanish. It differs from the previous `clinical_deidentificaiton` pipeline in that it includes the `ner_deid_subentity_augmented` NER model and some improvements in ContextualParsers and RegexMatchers.

The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask, fake or obfuscate the following entities: `AGE`, `DATE`, `PROFESSION`, `EMAIL`, `USERNAME`, `STREET`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `PATIENT`, `URL`, `MEDICALRECORD`, `IDNUM`, `ORGANIZATION`, `PHONE`, `ZIP`, `ACCOUNT`, `SSN`, `PLATE`, `SEX` and `IPADDR`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_augmented_es_3.4.1_3.0_1645005904505.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from johnsnowlabs import *

deid_pipeline = PretrainedPipeline("clinical_deidentification_augmented", "es", "clinical/models")

sample = """Datos del paciente.
Nombre:  Jose .
Apellidos: Aranda Martinez.
NHC: 2748903.
NASS: 26 37482910 04.
Domicilio: Calle Losada Martí 23, 5 B..
Localidad/ Provincia: Madrid.
CP: 28016.
Datos asistenciales.
Fecha de nacimiento: 15/04/1977.
País: España.
Edad: 37 años Sexo: F.
Fecha de Ingreso: 05/06/2018.
Médico: María Merino Viveros  NºCol: 28 28 35489.
Informe clínico del paciente: varón de 37 años con vida previa activa que refiere dolores osteoarticulares de localización variable en el último mes y fiebre en la última semana con picos (matutino y vespertino) de 40 C las últimas 24-48 horas, por lo que acude al Servicio de Urgencias. Antes de comenzar el cuadro estuvo en Extremadura en una región endémica de brucella, ingiriendo leche de cabra sin pasteurizar y queso de dicho ganado. Entre los comensales aparecieron varios casos de brucelosis. Durante el ingreso para estudio del síndrome febril con antecedentes epidemiológicos de posible exposición a Brucella presenta un cuadro de orquiepididimitis derecha.
La exploración física revela: Tª 40,2 C; T.A: 109/68 mmHg; Fc: 105 lpm. Se encuentra consciente, orientado, sudoroso, eupneico, con buen estado de nutrición e hidratación. En cabeza y cuello no se palpan adenopatías, ni bocio ni ingurgitación de vena yugular, con pulsos carotídeos simétricos. Auscultación cardíaca rítmica, sin soplos, roces ni extratonos. Auscultación pulmonar con conservación del murmullo vesicular. Abdomen blando, depresible, sin masas ni megalias. En la exploración neurológica no se detectan signos meníngeos ni datos de focalidad. Extremidades sin varices ni edemas. Pulsos periféricos presentes y simétricos. En la exploración urológica se aprecia el teste derecho aumentado de tamaño, no adherido a piel, con zonas de fluctuación e intensamente doloroso a la palpación, con pérdida del límite epidídimo-testicular y transiluminación positiva.
Los datos analíticos muestran los siguentes resultados: Hemograma: Hb 13,7 g/dl; leucocitos 14.610/mm3 (neutrófilos 77%); plaquetas 206.000/ mm3. VSG: 40 mm 1ª hora. Coagulación: TQ 87%; TTPA 25,8 seg. Bioquímica: Glucosa 117 mg/dl; urea 29 mg/dl; creatinina 0,9 mg/dl; sodio 136 mEq/l; potasio 3,6 mEq/l; GOT 11 U/l; GPT 24 U/l; GGT 34 U/l; fosfatasa alcalina 136 U/l; calcio 8,3 mg/dl. Orina: sedimento normal.
Durante el ingreso se solicitan Hemocultivos: positivo para Brucella y Serologías específicas para Brucella: Rosa de Bengala +++; Test de Coombs > 1/1280; Brucellacapt > 1/5120. Las pruebas de imagen solicitadas ( Rx tórax, Ecografía abdominal, TAC craneal, Ecocardiograma transtorácico) no evidencian patología significativa, excepto la Ecografía testicular, que muestra engrosamiento de la bolsa escrotal con pequeña cantidad de líquido con septos y testículo aumentado de tamaño con pequeñas zonas hipoecoicas en su interior que pueden representar microabscesos.
Con el diagnóstico de orquiepididimitis secundaria a Brucella se instaura tratamiento sintomático (antitérmicos, antiinflamatorios, reposo y elevación testicular) así como tratamiento antibiótico específico: Doxiciclina 100 mg vía oral cada 12 horas (durante 6 semanas) y Estreptomicina 1 gramo intramuscular cada 24 horas (durante 3 semanas). El paciente mejora significativamente de su cuadro tras una semana de ingreso, decidiéndose el alta a su domicilio donde completó la pauta de tratamiento antibiótico. En revisiones sucesivas en consultas se constató la completa remisión del cuadro.
Remitido por: Dra. María Merino Viveros Hospital Universitario de Getafe Servicio de Endocrinología y Nutrición Carretera de Toledo km 12,500 28905 Getafe - Madrid (España) Correo electrónico: marietta84@hotmail.com"""

result = deid_pipeline .annotate(sample)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val deid_pipeline = new PretrainedPipeline("clinical_deidentification_augmented", "es", "clinical/models")

sample = "Datos del paciente.
Nombre:  Jose .
Apellidos: Aranda Martinez.
NHC: 2748903.
NASS: 26 37482910 04.
Domicilio: Calle Losada Martí 23, 5 B..
Localidad/ Provincia: Madrid.
CP: 28016.
Datos asistenciales.
Fecha de nacimiento: 15/04/1977.
País: España.
Edad: 37 años Sexo: F.
Fecha de Ingreso: 05/06/2018.
Médico: María Merino Viveros  NºCol: 28 28 35489.
Informe clínico del paciente: varón de 37 años con vida previa activa que refiere dolores osteoarticulares de localización variable en el último mes y fiebre en la última semana con picos (matutino y vespertino) de 40 C las últimas 24-48 horas, por lo que acude al Servicio de Urgencias. Antes de comenzar el cuadro estuvo en Extremadura en una región endémica de brucella, ingiriendo leche de cabra sin pasteurizar y queso de dicho ganado. Entre los comensales aparecieron varios casos de brucelosis. Durante el ingreso para estudio del síndrome febril con antecedentes epidemiológicos de posible exposición a Brucella presenta un cuadro de orquiepididimitis derecha.
La exploración física revela: Tª 40,2 C; T.A: 109/68 mmHg; Fc: 105 lpm. Se encuentra consciente, orientado, sudoroso, eupneico, con buen estado de nutrición e hidratación. En cabeza y cuello no se palpan adenopatías, ni bocio ni ingurgitación de vena yugular, con pulsos carotídeos simétricos. Auscultación cardíaca rítmica, sin soplos, roces ni extratonos. Auscultación pulmonar con conservación del murmullo vesicular. Abdomen blando, depresible, sin masas ni megalias. En la exploración neurológica no se detectan signos meníngeos ni datos de focalidad. Extremidades sin varices ni edemas. Pulsos periféricos presentes y simétricos. En la exploración urológica se aprecia el teste derecho aumentado de tamaño, no adherido a piel, con zonas de fluctuación e intensamente doloroso a la palpación, con pérdida del límite epidídimo-testicular y transiluminación positiva.
Los datos analíticos muestran los siguentes resultados: Hemograma: Hb 13,7 g/dl; leucocitos 14.610/mm3 (neutrófilos 77%); plaquetas 206.000/ mm3. VSG: 40 mm 1ª hora. Coagulación: TQ 87%; TTPA 25,8 seg. Bioquímica: Glucosa 117 mg/dl; urea 29 mg/dl; creatinina 0,9 mg/dl; sodio 136 mEq/l; potasio 3,6 mEq/l; GOT 11 U/l; GPT 24 U/l; GGT 34 U/l; fosfatasa alcalina 136 U/l; calcio 8,3 mg/dl. Orina: sedimento normal.
Durante el ingreso se solicitan Hemocultivos: positivo para Brucella y Serologías específicas para Brucella: Rosa de Bengala +++; Test de Coombs > 1/1280; Brucellacapt > 1/5120. Las pruebas de imagen solicitadas ( Rx tórax, Ecografía abdominal, TAC craneal, Ecocardiograma transtorácico) no evidencian patología significativa, excepto la Ecografía testicular, que muestra engrosamiento de la bolsa escrotal con pequeña cantidad de líquido con septos y testículo aumentado de tamaño con pequeñas zonas hipoecoicas en su interior que pueden representar microabscesos.
Con el diagnóstico de orquiepididimitis secundaria a Brucella se instaura tratamiento sintomático (antitérmicos, antiinflamatorios, reposo y elevación testicular) así como tratamiento antibiótico específico: Doxiciclina 100 mg vía oral cada 12 horas (durante 6 semanas) y Estreptomicina 1 gramo intramuscular cada 24 horas (durante 3 semanas). El paciente mejora significativamente de su cuadro tras una semana de ingreso, decidiéndose el alta a su domicilio donde completó la pauta de tratamiento antibiótico. En revisiones sucesivas en consultas se constató la completa remisión del cuadro.
Remitido por: Dra. María Merino Viveros Hospital Universitario de Getafe Servicio de Endocrinología y Nutrición Carretera de Toledo km 12,500 28905 Getafe - Madrid (España) Correo electrónico: marietta84@hotmail.com
"

val result = deid_pipeline.annotate(sample)
```
</div>

## Results

```bash
Masked with entity labels
------------------------------
Datos <SEX>.
Nombre:  <PATIENT> .
Apellidos: <PATIENT>.
NHC: <SSN>.
NASS: <ID>.
Domicilio: <STREET>, <AGE> B..
Localidad/ Provincia: <CITY>.
CP: <ZIP>.
Datos asistenciales.
Fecha de nacimiento: <DATE>.
País: <COUNTRY>.
Edad: <AGE> años Sexo: <SEX>.
Fecha de Ingreso: <DATE>.
Médico: <DOCTOR>  NºCol: <ID>.
Informe clínico <SEX>: <SSN> de <AGE> años con vida previa activa que refiere dolores osteoarticulares de localización variable en el último mes y fiebre en la última semana con picos (matutino y vespertino) de 40 C las últimas 24-48 horas, por lo que acude al Servicio de Urgencias.
Antes de comenzar el cuadro estuvo en <CITY> en una región endémica de brucella, ingiriendo leche de cabra sin pasteurizar y queso de dicho ganado.
Entre los comensales aparecieron varios casos de brucelosis.
Durante el ingreso para estudio del síndrome febril con antecedentes epidemiológicos de posible exposición a Brucella presenta un cuadro de orquiepididimitis derecha.
La exploración física revela: Tª 40,2 C; T.A: 109/68 mmHg; Fc: 105 lpm. Se encuentra consciente, orientado, sudoroso, eupneico, con buen estado de nutrición e hidratación.
En cabeza y cuello no se palpan adenopatías, ni bocio ni ingurgitación de vena yugular, con pulsos carotídeos simétricos.
Auscultación cardíaca rítmica, sin soplos, roces ni extratonos.
Auscultación pulmonar con conservación del murmullo vesicular.
Abdomen blando, depresible, sin masas ni megalias.
En la exploración neurológica no se detectan signos meníngeos ni datos de focalidad.
Extremidades sin varices ni edemas.
Pulsos periféricos presentes y simétricos.
En la exploración urológica se aprecia el teste derecho aumentado de tamaño, no adherido a piel, con zonas de fluctuación e intensamente doloroso a la palpación, con pérdida del límite epidídimo-testicular y transiluminación positiva.
Los datos analíticos muestran los siguentes resultados: Hemograma: Hb 13,7 g/dl; leucocitos 14.610/mm3 (neutrófilos 77%); plaquetas 206.000/ mm3.
VSG: 40 mm 1ª hora.
Coagulación: TQ 87%;
<ORGANIZATION> 25,8 seg.
Bioquímica: Glucosa 117 mg/dl; urea 29 mg/dl; creatinina 0,9 mg/dl; sodio 136 mEq/l; potasio 3,6 mEq/l; GOT 11 U/l; GPT 24 U/l; GGT 34 U/l; fosfatasa alcalina 136 U/l; calcio 8,3 mg/dl.
Orina: sedimento normal.
Durante el ingreso se solicitan Hemocultivos: positivo para Brucella y Serologías específicas para Brucella: <PATIENT> +++;
Test de Coombs > <DATE>; Brucellacapt > 1/5120.
Las pruebas de imagen solicitadas ( Rx tórax, Ecografía abdominal, TAC craneal, Ecocardiograma transtorácico) no evidencian patología significativa, excepto la Ecografía testicular, que muestra engrosamiento de la bolsa escrotal con pequeña cantidad de líquido con septos y testículo aumentado de tamaño con pequeñas zonas hipoecoicas en su interior que pueden representar microabscesos.
Con el diagnóstico de orquiepididimitis secundaria a Brucella se instaura tratamiento sintomático (antitérmicos, antiinflamatorios, reposo y elevación testicular) así como tratamiento antibiótico específico: Doxiciclina 100 mg vía oral cada 12 horas (durante 6 semanas) y Estreptomicina 1 gramo intramuscular cada 24 horas (durante 3 semanas).
<SEX> <SSN> significativamente de su cuadro tras una semana de ingreso, decidiéndose el alta a su domicilio donde completó la pauta de tratamiento antibiótico. En revisiones sucesivas en consultas se constató la completa remisión del cuadro.
Remitido por: Dra.
<DOCTOR> <HOSPITAL> Servicio de Endocrinología y Nutrición <STREET> <ZIP> <CITY> - <CITY> (<COUNTRY>) Correo electrónico: <EMAIL>

Masked with chars
------------------------------
Datos [**********].
Nombre:  [**] .
Apellidos: [*************].
NHC: [*****].
NASS: [************].
Domicilio: [*******************], * B..
Localidad/ Provincia: [****].
CP: [***].
Datos asistenciales.
Fecha de nacimiento: [********].
País: [****].
Edad: ** años Sexo: *.
Fecha de Ingreso: [********].
Médico: [******************]  NºCol: [*********].
Informe clínico [**********]: [***] de ** años con vida previa activa que refiere dolores osteoarticulares de localización variable en el último mes y fiebre en la última semana con picos (matutino y vespertino) de 40 C las últimas 24-48 horas, por lo que acude al Servicio de Urgencias.
Antes de comenzar el cuadro estuvo en [*********] en una región endémica de brucella, ingiriendo leche de cabra sin pasteurizar y queso de dicho ganado.
Entre los comensales aparecieron varios casos de brucelosis.
Durante el ingreso para estudio del síndrome febril con antecedentes epidemiológicos de posible exposición a Brucella presenta un cuadro de orquiepididimitis derecha.
La exploración física revela: Tª 40,2 C; T.A: 109/68 mmHg; Fc: 105 lpm. Se encuentra consciente, orientado, sudoroso, eupneico, con buen estado de nutrición e hidratación.
En cabeza y cuello no se palpan adenopatías, ni bocio ni ingurgitación de vena yugular, con pulsos carotídeos simétricos.
Auscultación cardíaca rítmica, sin soplos, roces ni extratonos.
Auscultación pulmonar con conservación del murmullo vesicular.
Abdomen blando, depresible, sin masas ni megalias.
En la exploración neurológica no se detectan signos meníngeos ni datos de focalidad.
Extremidades sin varices ni edemas.
Pulsos periféricos presentes y simétricos.
En la exploración urológica se aprecia el teste derecho aumentado de tamaño, no adherido a piel, con zonas de fluctuación e intensamente doloroso a la palpación, con pérdida del límite epidídimo-testicular y transiluminación positiva.
Los datos analíticos muestran los siguentes resultados: Hemograma: Hb 13,7 g/dl; leucocitos 14.610/mm3 (neutrófilos 77%); plaquetas 206.000/ mm3.
VSG: 40 mm 1ª hora.
Coagulación: TQ 87%;
[**] 25,8 seg.
Bioquímica: Glucosa 117 mg/dl; urea 29 mg/dl; creatinina 0,9 mg/dl; sodio 136 mEq/l; potasio 3,6 mEq/l; GOT 11 U/l; GPT 24 U/l; GGT 34 U/l; fosfatasa alcalina 136 U/l; calcio 8,3 mg/dl.
Orina: sedimento normal.
Durante el ingreso se solicitan Hemocultivos: positivo para Brucella y Serologías específicas para Brucella: [*************] +++;
Test de Coombs > [****]; Brucellacapt > 1/5120.
Las pruebas de imagen solicitadas ( Rx tórax, Ecografía abdominal, TAC craneal, Ecocardiograma transtorácico) no evidencian patología significativa, excepto la Ecografía testicular, que muestra engrosamiento de la bolsa escrotal con pequeña cantidad de líquido con septos y testículo aumentado de tamaño con pequeñas zonas hipoecoicas en su interior que pueden representar microabscesos.
Con el diagnóstico de orquiepididimitis secundaria a Brucella se instaura tratamiento sintomático (antitérmicos, antiinflamatorios, reposo y elevación testicular) así como tratamiento antibiótico específico: Doxiciclina 100 mg vía oral cada 12 horas (durante 6 semanas) y Estreptomicina 1 gramo intramuscular cada 24 horas (durante 3 semanas).
[*********] [****] significativamente de su cuadro tras una semana de ingreso, decidiéndose el alta a su domicilio donde completó la pauta de tratamiento antibiótico. En revisiones sucesivas en consultas se constató la completa remisión del cuadro.
Remitido por: Dra.
[******************] [******************************] Servicio de Endocrinología y Nutrición [***************************] [***] [****] - [****] ([****]) Correo electrónico: [********************]

Masked with fixed length chars
------------------------------
Datos ****.
Nombre:  **** .
Apellidos: ****.
NHC: ****.
NASS: ****.
Domicilio: ****, **** B..
Localidad/ Provincia: ****.
CP: ****.
Datos asistenciales.
Fecha de nacimiento: ****.
País: ****.
Edad: **** años Sexo: ****.
Fecha de Ingreso: ****.
Médico: ****  NºCol: ****.
Informe clínico ****: **** de **** años con vida previa activa que refiere dolores osteoarticulares de localización variable en el último mes y fiebre en la última semana con picos (matutino y vespertino) de 40 C las últimas 24-48 horas, por lo que acude al Servicio de Urgencias.
Antes de comenzar el cuadro estuvo en **** en una región endémica de brucella, ingiriendo leche de cabra sin pasteurizar y queso de dicho ganado.
Entre los comensales aparecieron varios casos de brucelosis.
Durante el ingreso para estudio del síndrome febril con antecedentes epidemiológicos de posible exposición a Brucella presenta un cuadro de orquiepididimitis derecha.
La exploración física revela: Tª 40,2 C; T.A: 109/68 mmHg; Fc: 105 lpm. Se encuentra consciente, orientado, sudoroso, eupneico, con buen estado de nutrición e hidratación.
En cabeza y cuello no se palpan adenopatías, ni bocio ni ingurgitación de vena yugular, con pulsos carotídeos simétricos.
Auscultación cardíaca rítmica, sin soplos, roces ni extratonos.
Auscultación pulmonar con conservación del murmullo vesicular.
Abdomen blando, depresible, sin masas ni megalias.
En la exploración neurológica no se detectan signos meníngeos ni datos de focalidad.
Extremidades sin varices ni edemas.
Pulsos periféricos presentes y simétricos.
En la exploración urológica se aprecia el teste derecho aumentado de tamaño, no adherido a piel, con zonas de fluctuación e intensamente doloroso a la palpación, con pérdida del límite epidídimo-testicular y transiluminación positiva.
Los datos analíticos muestran los siguentes resultados: Hemograma: Hb 13,7 g/dl; leucocitos 14.610/mm3 (neutrófilos 77%); plaquetas 206.000/ mm3.
VSG: 40 mm 1ª hora.
Coagulación: TQ 87%;
**** 25,8 seg.
Bioquímica: Glucosa 117 mg/dl; urea 29 mg/dl; creatinina 0,9 mg/dl; sodio 136 mEq/l; potasio 3,6 mEq/l; GOT 11 U/l; GPT 24 U/l; GGT 34 U/l; fosfatasa alcalina 136 U/l; calcio 8,3 mg/dl.
Orina: sedimento normal.
Durante el ingreso se solicitan Hemocultivos: positivo para Brucella y Serologías específicas para Brucella: **** +++;
Test de Coombs > ****; Brucellacapt > 1/5120.
Las pruebas de imagen solicitadas ( Rx tórax, Ecografía abdominal, TAC craneal, Ecocardiograma transtorácico) no evidencian patología significativa, excepto la Ecografía testicular, que muestra engrosamiento de la bolsa escrotal con pequeña cantidad de líquido con septos y testículo aumentado de tamaño con pequeñas zonas hipoecoicas en su interior que pueden representar microabscesos.
Con el diagnóstico de orquiepididimitis secundaria a Brucella se instaura tratamiento sintomático (antitérmicos, antiinflamatorios, reposo y elevación testicular) así como tratamiento antibiótico específico: Doxiciclina 100 mg vía oral cada 12 horas (durante 6 semanas) y Estreptomicina 1 gramo intramuscular cada 24 horas (durante 3 semanas).
**** **** significativamente de su cuadro tras una semana de ingreso, decidiéndose el alta a su domicilio donde completó la pauta de tratamiento antibiótico. En revisiones sucesivas en consultas se constató la completa remisión del cuadro.
Remitido por: Dra.
**** **** Servicio de Endocrinología y Nutrición **** **** **** - **** (****) Correo electrónico: ****

Obfuscated
------------------------------
Datos Hombre.
Nombre:  Aurora Garrido Paez .
Apellidos: Aurora Garrido Paez.
NHC: BBBBBBBBQR648597.
NASS: 48127833R.
Domicilio: C/ Rambla, 246, 5 B..
Localidad/ Provincia: Alicante.
CP: 24202.
Datos asistenciales.
Fecha de nacimiento: 21/04/1977.
País: Portugal.
Edad: 56 años Sexo: Hombre.
Fecha de Ingreso: 10/07/2018.
Médico: Francisco José Roca Bermúdez  NºCol: 21344083-P.
Informe clínico Hombre: 041010000011 de 56 años con vida previa activa que refiere dolores osteoarticulares de localización variable en el último mes y fiebre en la última semana con picos (matutino y vespertino) de 40 C las últimas 24-48 horas, por lo que acude al Servicio de Urgencias.
Antes de comenzar el cuadro estuvo en Zaragoza en una región endémica de brucella, ingiriendo leche de cabra sin pasteurizar y queso de dicho ganado.
Entre los comensales aparecieron varios casos de brucelosis.
Durante el ingreso para estudio del síndrome febril con antecedentes epidemiológicos de posible exposición a Brucella presenta un cuadro de orquiepididimitis derecha.
La exploración física revela: Tª 40,2 C; T.A: 109/68 mmHg; Fc: 105 lpm. Se encuentra consciente, orientado, sudoroso, eupneico, con buen estado de nutrición e hidratación.
En cabeza y cuello no se palpan adenopatías, ni bocio ni ingurgitación de vena yugular, con pulsos carotídeos simétricos.
Auscultación cardíaca rítmica, sin soplos, roces ni extratonos.
Auscultación pulmonar con conservación del murmullo vesicular.
Abdomen blando, depresible, sin masas ni megalias.
En la exploración neurológica no se detectan signos meníngeos ni datos de focalidad.
Extremidades sin varices ni edemas.
Pulsos periféricos presentes y simétricos.
En la exploración urológica se aprecia el teste derecho aumentado de tamaño, no adherido a piel, con zonas de fluctuación e intensamente doloroso a la palpación, con pérdida del límite epidídimo-testicular y transiluminación positiva.
Los datos analíticos muestran los siguentes resultados: Hemograma: Hb 13,7 g/dl; leucocitos 14.610/mm3 (neutrófilos 77%); plaquetas 206.000/ mm3.
VSG: 40 mm 1ª hora.
Coagulación: TQ 87%;
Tecnogroup SL 25,8 seg.
Bioquímica: Glucosa 117 mg/dl; urea 29 mg/dl; creatinina 0,9 mg/dl; sodio 136 mEq/l; potasio 3,6 mEq/l; GOT 11 U/l; GPT 24 U/l; GGT 34 U/l; fosfatasa alcalina 136 U/l; calcio 8,3 mg/dl.
Orina: sedimento normal.
Durante el ingreso se solicitan Hemocultivos: positivo para Brucella y Serologías específicas para Brucella: María Miguélez Sanz +++;
Test de Coombs > 07-25-1974; Brucellacapt > 1/5120.
Las pruebas de imagen solicitadas ( Rx tórax, Ecografía abdominal, TAC craneal, Ecocardiograma transtorácico) no evidencian patología significativa, excepto la Ecografía testicular, que muestra engrosamiento de la bolsa escrotal con pequeña cantidad de líquido con septos y testículo aumentado de tamaño con pequeñas zonas hipoecoicas en su interior que pueden representar microabscesos.
Con el diagnóstico de orquiepididimitis secundaria a Brucella se instaura tratamiento sintomático (antitérmicos, antiinflamatorios, reposo y elevación testicular) así como tratamiento antibiótico específico: Doxiciclina 100 mg vía oral cada 12 horas (durante 6 semanas) y Estreptomicina 1 gramo intramuscular cada 24 horas (durante 3 semanas).
F. 041010000011 significativamente de su cuadro tras una semana de ingreso, decidiéndose el alta a su domicilio donde completó la pauta de tratamiento antibiótico. En revisiones sucesivas en consultas se constató la completa remisión del cuadro.
Remitido por: Dra.
Francisco José Roca Bermúdez Hospital 12 de Octubre Servicio de Endocrinología y Nutrición Calle Ramón y Cajal s/n 03129 Zaragoza - Alicante (Portugal) Correo electrónico: richard@company.it
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification_augmented|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|281.3 MB|

## Included Models

- nlp.DocumentAssembler
- nlp.SentenceDetectorDLModel
- nlp.TokenizerModel
- nlp.WordEmbeddingsModel
- medical.NerModel
- nlp.NerConverter
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- nlp.RegexMatcherModel
- ChunkMergeModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- medical.DeIdentificationModel
- Finisher
