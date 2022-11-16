---
layout: model
title: Clinical Deidentification (French)
author: John Snow Labs
name: clinical_deidentification
date: 2022-02-23
tags: [deid, fr, licensed]
task: De-identification
language: fr
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pipeline can be used to deidentify PHI information from medical texts in French. The pipeline can mask and obfuscate the following entities: `DATE`, `AGE`, `SEX`, `PROFESSION`, `ORGANIZATION`, `PHONE`, `E-MAIL`, `ZIP`, `STREET`, `CITY`, `COUNTRY`, `PATIENT`, `DOCTOR`, `HOSPITAL`, `MEDICALRECORD`, `SSN`, `IDNUM`, `ACCOUNT`, `PLATE`, `USERNAME`, `URL`, and `IPADDR`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/clinical_deidentification_fr_3.4.1_3.0_1645643868811.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "fr", "clinical/models")

sample = """COMPTE-RENDU D'HOSPITALISATION
PRENOM : Jean
NOM : Dubois
NUMÉRO DE SÉCURITÉ SOCIALE : 1780160471058
ADRESSE : 18 Avenue Matabiau
VILLE : Grenoble
CODE POSTAL : 38000
DATE DE NAISSANCE : 03/03/1946
Âge : 70 ans 
Sexe : H
COURRIEL : jdubois@hotmail.fr
DATE D'ADMISSION : 12/12/2016
MÉDÉCIN : Dr Michel Renaud
RAPPORT CLINIQUE : 70 ans, retraité, sans allergie médicamenteuse connue, qui présente comme antécédents : ancien accident du travail avec fractures vertébrales et des côtes ; opéré de la maladie de Dupuytren à la main droite et d'un pontage ilio-fémoral gauche ; diabète de type II, hypercholestérolémie et hyperuricémie ; alcoolisme actif, fume 20 cigarettes / jour.
Il nous a été adressé car il présentait une hématurie macroscopique postmictionnelle à une occasion et une microhématurie persistante par la suite, avec une miction normale.
L'examen physique a montré un bon état général, avec un abdomen et des organes génitaux normaux ; le toucher rectal était compatible avec un adénome de la prostate de grade I/IV.
L'analyse d'urine a montré 4 globules rouges/champ et 0-5 leucocytes/champ ; le reste du sédiment était normal.
Hémogramme normal ; la biochimie a montré une glycémie de 169 mg/dl et des triglycérides de 456 mg/dl ; les fonctions hépatiques et rénales étaient normales. PSA de 1,16 ng/ml.
ADDRESSÉ À : Dre Marie Breton - Centre Hospitalier de Bellevue Service D'Endocrinologie et de Nutrition - Rue Paulin Bussières, 38000 Grenoble
COURRIEL : mariebreton@chb.fr
"""

result = deid_pipeline.annotate(sample)
```
```scala
val deid_pipeline = new PretrainedPipeline("clinical_deidentification", "fr", "clinical/models")

sample = "COMPTE-RENDU D'HOSPITALISATION
PRENOM : Jean
NOM : Dubois
NUMÉRO DE SÉCURITÉ SOCIALE : 1780160471058
ADRESSE : 18 Avenue Matabiau
VILLE : Grenoble
CODE POSTAL : 38000
DATE DE NAISSANCE : 03/03/1946
Âge : 70 ans 
Sexe : H
COURRIEL : jdubois@hotmail.fr
DATE D'ADMISSION : 12/12/2016
MÉDÉCIN : Dr Michel Renaud
RAPPORT CLINIQUE : 70 ans, retraité, sans allergie médicamenteuse connue, qui présente comme antécédents : ancien accident du travail avec fractures vertébrales et des côtes ; opéré de la maladie de Dupuytren à la main droite et d'un pontage ilio-fémoral gauche ; diabète de type II, hypercholestérolémie et hyperuricémie ; alcoolisme actif, fume 20 cigarettes / jour.
Il nous a été adressé car il présentait une hématurie macroscopique postmictionnelle à une occasion et une microhématurie persistante par la suite, avec une miction normale.
L'examen physique a montré un bon état général, avec un abdomen et des organes génitaux normaux ; le toucher rectal était compatible avec un adénome de la prostate de grade I/IV.
L'analyse d'urine a montré 4 globules rouges/champ et 0-5 leucocytes/champ ; le reste du sédiment était normal.
Hémogramme normal ; la biochimie a montré une glycémie de 169 mg/dl et des triglycérides de 456 mg/dl ; les fonctions hépatiques et rénales étaient normales. PSA de 1,16 ng/ml.
ADDRESSÉ À : Dre Marie Breton - Centre Hospitalier de Bellevue Service D'Endocrinologie et de Nutrition - Rue Paulin Bussières, 38000 Grenoble
COURRIEL : mariebreton@chb.fr
"

val result = deid_pipeline.annotate(sample)
```


{:.nlu-block}
```python
import nlu
nlu.load("fr.deid_obfuscated").predict("""COMPTE-RENDU D'HOSPITALISATION
PRENOM : Jean
NOM : Dubois
NUMÉRO DE SÉCURITÉ SOCIALE : 1780160471058
ADRESSE : 18 Avenue Matabiau
VILLE : Grenoble
CODE POSTAL : 38000
DATE DE NAISSANCE : 03/03/1946
Âge : 70 ans 
Sexe : H
COURRIEL : jdubois@hotmail.fr
DATE D'ADMISSION : 12/12/2016
MÉDÉCIN : Dr Michel Renaud
RAPPORT CLINIQUE : 70 ans, retraité, sans allergie médicamenteuse connue, qui présente comme antécédents : ancien accident du travail avec fractures vertébrales et des côtes ; opéré de la maladie de Dupuytren à la main droite et d'un pontage ilio-fémoral gauche ; diabète de type II, hypercholestérolémie et hyperuricémie ; alcoolisme actif, fume 20 cigarettes / jour.
Il nous a été adressé car il présentait une hématurie macroscopique postmictionnelle à une occasion et une microhématurie persistante par la suite, avec une miction normale.
L'examen physique a montré un bon état général, avec un abdomen et des organes génitaux normaux ; le toucher rectal était compatible avec un adénome de la prostate de grade I/IV.
L'analyse d'urine a montré 4 globules rouges/champ et 0-5 leucocytes/champ ; le reste du sédiment était normal.
Hémogramme normal ; la biochimie a montré une glycémie de 169 mg/dl et des triglycérides de 456 mg/dl ; les fonctions hépatiques et rénales étaient normales. PSA de 1,16 ng/ml.
ADDRESSÉ À : Dre Marie Breton - Centre Hospitalier de Bellevue Service D'Endocrinologie et de Nutrition - Rue Paulin Bussières, 38000 Grenoble
COURRIEL : mariebreton@chb.fr
""")
```

</div>

## Results

```bash
Masked with entity labels
------------------------------
COMPTE-RENDU D'HOSPITALISATION
PRENOM : <PATIENT>
NOM : <PATIENT>
NUMÉRO DE SÉCURITÉ SOCIALE : <SSN>
ADRESSE : <STREET>
VILLE : <CITY>
CODE POSTAL : <ZIP>
DATE DE NAISSANCE : <DATE>
Âge : <AGE> 
Sexe : <SEX>
COURRIEL : <E-MAIL>
DATE D'ADMISSION : <DATE>
MÉDÉCIN : <DOCTOR>
RAPPORT CLINIQUE : <AGE> ans, retraité, sans allergie médicamenteuse connue, qui présente comme antécédents : ancien accident du travail avec fractures vertébrales et des côtes ; opéré de la maladie de Dupuytren à la main droite et d'un pontage ilio-fémoral gauche ; diabète de type II, hypercholestérolémie et hyperuricémie ; alcoolisme actif, fume 20 cigarettes / jour.
<SEX> nous a été adressé car <SEX> présentait une hématurie macroscopique postmictionnelle à une occasion et une microhématurie persistante par la suite, avec une miction normale.
L'examen physique a montré un bon état général, avec un abdomen et des organes génitaux normaux ; le toucher rectal était compatible avec un adénome de la prostate de grade I/IV.
L'analyse d'urine a montré 4 globules rouges/champ et 0-5 leucocytes/champ ; le reste du sédiment était normal.
Hémogramme normal ; la biochimie a montré une glycémie de 169 mg/dl et des triglycérides de 456 mg/dl ; les fonctions hépatiques et rénales étaient normales.
PSA de 1,16 ng/ml.
ADDRESSÉ À : <DOCTOR> - <HOSPITAL> Service D'Endocrinologie et de Nutrition - <STREET>, <ZIP> <CITY>
COURRIEL : <E-MAIL>


Masked with chars
------------------------------
COMPTE-RENDU D'HOSPITALISATION
PRENOM : [**]
NOM : [****]
NUMÉRO DE SÉCURITÉ SOCIALE : [***********]
ADRESSE : [****************]
VILLE : [******]
CODE POSTAL : [***]
DATE DE NAISSANCE : [********]
Âge : [****] 
Sexe : *
COURRIEL : [****************]
DATE D'ADMISSION : [********]
MÉDÉCIN : [**************]
RAPPORT CLINIQUE : ** ans, retraité, sans allergie médicamenteuse connue, qui présente comme antécédents : ancien accident du travail avec fractures vertébrales et des côtes ; opéré de la maladie de Dupuytren à la main droite et d'un pontage ilio-fémoral gauche ; diabète de type II, hypercholestérolémie et hyperuricémie ; alcoolisme actif, fume 20 cigarettes / jour.
** nous a été adressé car ** présentait une hématurie macroscopique postmictionnelle à une occasion et une microhématurie persistante par la suite, avec une miction normale.
L'examen physique a montré un bon état général, avec un abdomen et des organes génitaux normaux ; le toucher rectal était compatible avec un adénome de la prostate de grade I/IV.
L'analyse d'urine a montré 4 globules rouges/champ et 0-5 leucocytes/champ ; le reste du sédiment était normal.
Hémogramme normal ; la biochimie a montré une glycémie de 169 mg/dl et des triglycérides de 456 mg/dl ; les fonctions hépatiques et rénales étaient normales.
PSA de 1,16 ng/ml.
ADDRESSÉ À : [**************] - [****************************] Service D'Endocrinologie et de Nutrition - [******************], [***] [******]
COURRIEL : [****************]


Masked with fixed length chars
------------------------------
COMPTE-RENDU D'HOSPITALISATION
PRENOM : ****
NOM : ****
NUMÉRO DE SÉCURITÉ SOCIALE : ****
ADRESSE : ****
VILLE : ****
CODE POSTAL : ****
DATE DE NAISSANCE : ****
Âge : **** 
Sexe : ****
COURRIEL : ****
DATE D'ADMISSION : ****
MÉDÉCIN : ****
RAPPORT CLINIQUE : **** ans, retraité, sans allergie médicamenteuse connue, qui présente comme antécédents : ancien accident du travail avec fractures vertébrales et des côtes ; opéré de la maladie de Dupuytren à la main droite et d'un pontage ilio-fémoral gauche ; diabète de type II, hypercholestérolémie et hyperuricémie ; alcoolisme actif, fume 20 cigarettes / jour.
**** nous a été adressé car **** présentait une hématurie macroscopique postmictionnelle à une occasion et une microhématurie persistante par la suite, avec une miction normale.
L'examen physique a montré un bon état général, avec un abdomen et des organes génitaux normaux ; le toucher rectal était compatible avec un adénome de la prostate de grade I/IV.
L'analyse d'urine a montré 4 globules rouges/champ et 0-5 leucocytes/champ ; le reste du sédiment était normal.
Hémogramme normal ; la biochimie a montré une glycémie de 169 mg/dl et des triglycérides de 456 mg/dl ; les fonctions hépatiques et rénales étaient normales.
PSA de 1,16 ng/ml.
ADDRESSÉ À : **** - **** Service D'Endocrinologie et de Nutrition - ****, **** ****
COURRIEL : ****


Obfuscated
------------------------------
COMPTE-RENDU D'HOSPITALISATION
PRENOM : Mme Ollivier
NOM : Mme Traore
NUMÉRO DE SÉCURITÉ SOCIALE : 164033818514436
ADRESSE : 731, boulevard de Legrand
VILLE : Sainte Antoine
CODE POSTAL : 37443
DATE DE NAISSANCE : 18/03/1946
Âge : 46 
Sexe : Femme
COURRIEL : georgeslemonnier@live.com
DATE D'ADMISSION : 10/01/2017
MÉDÉCIN : Pr. Manon Dupuy
RAPPORT CLINIQUE : 26 ans, retraité, sans allergie médicamenteuse connue, qui présente comme antécédents : ancien accident du travail avec fractures vertébrales et des côtes ; opéré de la maladie de Dupuytren à la main droite et d'un pontage ilio-fémoral gauche ; diabète de type II, hypercholestérolémie et hyperuricémie ; alcoolisme actif, fume 20 cigarettes / jour.
Homme nous a été adressé car Homme présentait une hématurie macroscopique postmictionnelle à une occasion et une microhématurie persistante par la suite, avec une miction normale.
L'examen physique a montré un bon état général, avec un abdomen et des organes génitaux normaux ; le toucher rectal était compatible avec un adénome de la prostate de grade I/IV.
L'analyse d'urine a montré 4 globules rouges/champ et 0-5 leucocytes/champ ; le reste du sédiment était normal.
Hémogramme normal ; la biochimie a montré une glycémie de 169 mg/dl et des triglycérides de 456 mg/dl ; les fonctions hépatiques et rénales étaient normales.
PSA de 1,16 ng/ml.
ADDRESSÉ À : Dr Tristan-Gilbert Poulain - CENTRE HOSPITALIER D'ORTHEZ Service D'Endocrinologie et de Nutrition - 6, avenue Pages, 37443 Sainte Antoine
COURRIEL : massecatherine@bouygtel.fr
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|clinical_deidentification|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|fr|
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
- ContextualParserModel
- ContextualParserModel
- RegexMatcherModel
- RegexMatcherModel
- ChunkMergeModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
- Finisher