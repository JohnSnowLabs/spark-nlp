---
layout: model
title: Clinical Deidentification (Italian)
author: John Snow Labs
name: clinical_deidentification
date: 2022-03-28
tags: [deidentification, pipeline, it, licensed]
task: De-identification
language: it
edition: Healthcare NLP 3.4.2
spark_version: 2.4
supported: true
recommended: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline can be used to deidentify PHI information from medical texts in Italian. The pipeline can mask and obfuscate the following entities: `DATE`, `AGE`, `SEX`, `PROFESSION`, `ORGANIZATION`, `PHONE`, `E-MAIL`, `ZIP`, `STREET`, `CITY`, `COUNTRY`, `PATIENT`, `DOCTOR`, `HOSPITAL`, `MEDICALRECORD`, `SSN`, `IDNUM`, `ACCOUNT`, `PLATE`, `USERNAME`, `URL`, and `IPADDR`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_it_3.4.2_2.4_1648498695375.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_it_3.4.2_2.4_1648498695375.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "it", "clinical/models")

sample = """RAPPORTO DI RICOVERO
NOME: Lodovico Fibonacci
CODICE FISCALE: MVANSK92F09W408A
INDIRIZZO: Viale Burcardo 7
CITTÀ : Napoli
CODICE POSTALE: 80139
DATA DI NASCITA: 03/03/1946
ETÀ: 70 anni 
SESSO: M
EMAIL: lpizzo@tim.it
DATA DI AMMISSIONE: 12/12/2016
DOTTORE: Eva Viviani
RAPPORTO CLINICO: 70 anni, pensionato, senza allergie farmacologiche note, che presenta la seguente storia: ex incidente sul lavoro con fratture vertebrali e costali; operato per la malattia di Dupuytren alla mano destra e un bypass ileo-femorale sinistro; diabete di tipo II, ipercolesterolemia e iperuricemia; alcolismo attivo, fuma 20 sigarette/giorno.
È stato indirizzato a noi perché ha presentato un'ematuria macroscopica post-evacuazione in un'occasione e una microematuria persistente in seguito, con un'evacuazione normale.
L'esame fisico ha mostrato buone condizioni generali, con addome e genitali normali; l'esame digitale rettale era coerente con un adenoma prostatico di grado I/IV.
L'analisi delle urine ha mostrato 4 globuli rossi/campo e 0-5 leucociti/campo; il resto del sedimento era normale.
L'emocromo è normale; la biochimica ha mostrato una glicemia di 169 mg/dl e trigliceridi 456 mg/dl; la funzione epatica e renale sono normali. PSA di 1,16 ng/ml.

INDIRIZZATO A: Dott. Bruno Ferrabosco - ASL Napoli 1 Centro, Dipartimento di Endocrinologia e Nutrizione - Stretto Scamarcio 320, 80138 Napoli
EMAIL: bferrabosco@poste.it"""

result = deid_pipeline.annotate(sample)
```
```scala
val deid_pipeline = new PretrainedPipeline("clinical_deidentification", "it", "clinical/models")

sample = "RAPPORTO DI RICOVERO
NOME: Lodovico Fibonacci
CODICE FISCALE: MVANSK92F09W408A
INDIRIZZO: Viale Burcardo 7
CITTÀ : Napoli
CODICE POSTALE: 80139
DATA DI NASCITA: 03/03/1946
ETÀ: 70 anni 
SESSO: M
EMAIL: lpizzo@tim.it
DATA DI AMMISSIONE: 12/12/2016
DOTTORE: Eva Viviani
RAPPORTO CLINICO: 70 anni, pensionato, senza allergie farmacologiche note, che presenta la seguente storia: ex incidente sul lavoro con fratture vertebrali e costali; operato per la malattia di Dupuytren alla mano destra e un bypass ileo-femorale sinistro; diabete di tipo II, ipercolesterolemia e iperuricemia; alcolismo attivo, fuma 20 sigarette/giorno.
È stato indirizzato a noi perché ha presentato un'ematuria macroscopica post-evacuazione in un'occasione e una microematuria persistente in seguito, con un'evacuazione normale.
L'esame fisico ha mostrato buone condizioni generali, con addome e genitali normali; l'esame digitale rettale era coerente con un adenoma prostatico di grado I/IV.
L'analisi delle urine ha mostrato 4 globuli rossi/campo e 0-5 leucociti/campo; il resto del sedimento era normale.
L'emocromo è normale; la biochimica ha mostrato una glicemia di 169 mg/dl e trigliceridi 456 mg/dl; la funzione epatica e renale sono normali. PSA di 1,16 ng/ml.

INDIRIZZATO A: Dott. Bruno Ferrabosco - ASL Napoli 1 Centro, Dipartimento di Endocrinologia e Nutrizione - Stretto Scamarcio 320, 80138 Napoli
EMAIL: bferrabosco@poste.it"

val result = deid_pipeline.annotate(sample)
```
</div>

## Results

```bash
Masked with entity labels
------------------------------
RAPPORTO DI RICOVERO
NOME: <PATIENT>
CODICE FISCALE: <SSN>
INDIRIZZO: <STREET>
CITTÀ : <CITY>
CODICE POSTALE: <ZIP>
DATA DI NASCITA: <DATE>
ETÀ: <AGE> anni 
SESSO: <SEX>
EMAIL: <E-MAIL>
DATA DI AMMISSIONE: <DATE>
DOTTORE: <DOCTOR>
RAPPORTO CLINICO: <AGE> anni, pensionato, senza allergie farmacologiche note, che presenta la seguente storia: ex incidente sul lavoro con fratture vertebrali e costali; operato per la malattia di Dupuytren alla mano destra e un bypass ileo-femorale sinistro; diabete di tipo II, ipercolesterolemia e iperuricemia; alcolismo attivo, fuma 20 sigarette/giorno.
È stato indirizzato a noi perché ha presentato un'ematuria macroscopica post-evacuazione in un'occasione e una microematuria persistente in seguito, con un'evacuazione normale.
L'esame fisico ha mostrato buone condizioni generali, con addome e genitali normali; l'esame digitale rettale era coerente con un adenoma prostatico di grado I/IV.
L'analisi delle urine ha mostrato 4 globuli rossi/campo e 0-5 leucociti/campo; il resto del sedimento era normale.
L'emocromo è normale; la biochimica ha mostrato una glicemia di 169 mg/dl e trigliceridi 456 mg/dl; la funzione epatica e renale sono normali.
PSA di 1,16 ng/ml.
INDIRIZZATO A: Dott.
<DOCTOR> - <HOSPITAL>, Dipartimento di Endocrinologia e Nutrizione - <STREET>, <ZIP> <CITY>
EMAIL: <E-MAIL>


Masked with chars
------------------------------
RAPPORTO DI RICOVERO
NOME: [****************]
CODICE FISCALE: [**************]
INDIRIZZO: [**************]
CITTÀ : [****]
CODICE POSTALE: [***]DATA DI NASCITA: [********]
ETÀ: **anni 
SESSO: *
EMAIL: [***********]
DATA DI AMMISSIONE: [********]
DOTTORE: [*********]
RAPPORTO CLINICO: **anni, pensionato, senza allergie farmacologiche note, che presenta la seguente storia: ex incidente sul lavoro con fratture vertebrali e costali; operato per la malattia di Dupuytren alla mano destra e un bypass ileo-femorale sinistro; diabete di tipo II, ipercolesterolemia e iperuricemia; alcolismo attivo, fuma 20 sigarette/giorno.
È stato indirizzato a noi perché ha presentato un'ematuria macroscopica post-evacuazione in un'occasione e una microematuria persistente in seguito, con un'evacuazione normale.
L'esame fisico ha mostrato buone condizioni generali, con addome e genitali normali; l'esame digitale rettale era coerente con un adenoma prostatico di grado I/IV.
L'analisi delle urine ha mostrato 4 globuli rossi/campo e 0-5 leucociti/campo; il resto del sedimento era normale.
L'emocromo è normale; la biochimica ha mostrato una glicemia di 169 mg/dl e trigliceridi 456 mg/dl; la funzione epatica e renale sono normali.
PSA di 1,16 ng/ml.
INDIRIZZATO A: Dott.
[**************] - [*****************], Dipartimento di Endocrinologia e Nutrizione - [*******************], [***] [****]
EMAIL: [******************]


Masked with fixed length chars
------------------------------
RAPPORTO DI RICOVERO
NOME: ****
CODICE FISCALE: ****
INDIRIZZO: ****
CITTÀ : ****
CODICE POSTALE: ****DATA DI NASCITA: ****
ETÀ: **** anni 
SESSO: ****
EMAIL: ****
DATA DI AMMISSIONE: ****
DOTTORE: ****
RAPPORTO CLINICO: **** anni, pensionato, senza allergie farmacologiche note, che presenta la seguente storia: ex incidente sul lavoro con fratture vertebrali e costali; operato per la malattia di Dupuytren alla mano destra e un bypass ileo-femorale sinistro; diabete di tipo II, ipercolesterolemia e iperuricemia; alcolismo attivo, fuma 20 sigarette/giorno.
È stato indirizzato a noi perché ha presentato un'ematuria macroscopica post-evacuazione in un'occasione e una microematuria persistente in seguito, con un'evacuazione normale.
L'esame fisico ha mostrato buone condizioni generali, con addome e genitali normali; l'esame digitale rettale era coerente con un adenoma prostatico di grado I/IV.
L'analisi delle urine ha mostrato 4 globuli rossi/campo e 0-5 leucociti/campo; il resto del sedimento era normale.
L'emocromo è normale; la biochimica ha mostrato una glicemia di 169 mg/dl e trigliceridi 456 mg/dl; la funzione epatica e renale sono normali.
PSA di 1,16 ng/ml.
INDIRIZZATO A: Dott.
**** - ****, Dipartimento di Endocrinologia e Nutrizione - ****, **** ****
EMAIL: ****


Obfuscated
------------------------------
RAPPORTO DI RICOVERO
NOME: Scotto-Polani
CODICE FISCALE: ECI-QLN77G15L455Y
INDIRIZZO: Viale Orlando 808
CITTÀ : Sesto Raimondo
CODICE POSTALE: 53581DATA DI NASCITA: 09/03/1946
ETÀ: 5 anni 
SESSO: U
EMAIL: HenryWatson@world.com
DATA DI AMMISSIONE: 10/01/2017
DOTTORE: Sig. Fredo Marangoni
RAPPORTO CLINICO: 5 anni, pensionato, senza allergie farmacologiche note, che presenta la seguente storia: ex incidente sul lavoro con fratture vertebrali e costali; operato per la malattia di Dupuytren alla mano destra e un bypass ileo-femorale sinistro; diabete di tipo II, ipercolesterolemia e iperuricemia; alcolismo attivo, fuma 20 sigarette/giorno.
È stato indirizzato a noi perché ha presentato un'ematuria macroscopica post-evacuazione in un'occasione e una microematuria persistente in seguito, con un'evacuazione normale.
L'esame fisico ha mostrato buone condizioni generali, con addome e genitali normali; l'esame digitale rettale era coerente con un adenoma prostatico di grado I/IV.
L'analisi delle urine ha mostrato 4 globuli rossi/campo e 0-5 leucociti/campo; il resto del sedimento era normale.
L'emocromo è normale; la biochimica ha mostrato una glicemia di 169 mg/dl e trigliceridi 456 mg/dl; la funzione epatica e renale sono normali.
PSA di 1,16 ng/ml.
INDIRIZZATO A: Dott.
Antonio Rusticucci - ASL 7 DI CARBONIA AZIENDA U.S.L. N. 7, Dipartimento di Endocrinologia e Nutrizione - Via Giorgio 0 Appartamento 26, 03461 Sesto Raimondo
EMAIL: murat.g@jsl.com
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|it|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- ContextualParserModel
- ContextualParserModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- RegexMatcherModel
- ChunkMergeModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
- Finisher
