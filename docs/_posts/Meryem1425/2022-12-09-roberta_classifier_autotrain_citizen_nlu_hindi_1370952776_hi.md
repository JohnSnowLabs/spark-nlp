---
layout: model
title: Hindi RobertaForSequenceClassification Cased model (from neuralspace)
author: John Snow Labs
name: roberta_classifier_autotrain_citizen_nlu_hindi_1370952776
date: 2022-12-09
tags: [hi, open_source, roberta, sequence_classification, classification, tensorflow]
task: Text Classification
language: hi
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autotrain-citizen_nlu_hindi-1370952776` is a Hindi model originally trained by `neuralspace`.

## Predicted Entities

`ReportingMissingPets`, `EligibilityForBloodDonationCovidGap`, `ReportingPropertyTakeOver`, `IntentForBloodReceivalAppointment`, `EligibilityForBloodDonationSTD`, `InquiryForDoctorConsultation`, `InquiryOfCovidSymptoms`, `InquiryForVaccineCount`, `InquiryForCovidPrevention`, `InquiryForVaccinationRequirements`, `EligibilityForBloodDonationForPregnantWomen`, `ReportingCyberCrime`, `ReportingHitAndRun`, `ReportingTresspassing`, `InquiryofBloodDonationRequirements`, `ReportingMurder`, `ReportingVehicleAccident`, `ReportingMissingPerson`, `EligibilityForBloodDonationAgeLimit`, `ReportingAnimalPoaching`, `InquiryOfEmergencyContact`, `InquiryForQuarantinePeriod`, `ContactRealPerson`, `IntentForBloodDonationAppointment`, `ReportingMissingVehicle`, `InquiryForCovidRecentCasesCount`, `InquiryOfContact`, `StatusOfFIR`, `InquiryofVaccinationAgeLimit`, `InquiryForCovidTotalCasesCount`, `EligibilityForBloodDonationGap`, `InquiryofPostBloodDonationEffects`, `InquiryofPostBloodReceivalCareSchemes`, `EligibilityForBloodReceiversBloodGroup`, `EligitbilityForVaccine`, `InquiryOfLockdownDetails`, `ReportingSexualAssault`, `InquiryForVaccineCost`, `InquiryForCovidDeathCount`, `ReportingDrugConsumption`, `ReportingDrugTrafficing`, `InquiryofPostBloodDonationCertificate`, `ReportingDowry`, `ReportingChildAbuse`, `ReportingAnimalAbuse`, `InquiryofPostBloodReceivalEffects`, `Eligibility For BloodDonationWithComorbidities`, `InquiryOfTiming`, `InquiryForCovidActiveCasesCount`, `InquiryOfLocation`, `InquiryofPostBloodDonationCareSchemes`, `ReportingTheft`, `InquiryForTravelRestrictions`, `ReportingDomesticViolence`, `InquiryofBloodReceivalRequirements`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_autotrain_citizen_nlu_hindi_1370952776_hi_4.2.4_3.0_1670623674646.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_autotrain_citizen_nlu_hindi_1370952776","hi") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, roberta_classifier])

data = spark.createDataFrame([["I love you!"], ["I feel lucky to be here."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols("text")
    .setOutputCols("document")
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_autotrain_citizen_nlu_hindi_1370952776","hi") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, roberta_classifier))

val data = Seq("I love you!").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_autotrain_citizen_nlu_hindi_1370952776|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|hi|
|Size:|314.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/neuralspace/autotrain-citizen_nlu_hindi-1370952776