---
layout: model
title: Dutch BertForSequenceClassification Cased model (from Jeska)
author: John Snow Labs
name: bert_classifier_vaccinchatsentenceclassifierdutch_frombertje2_dadialogqonly
date: 2022-09-19
tags: [bert, sequence_classification, classification, open_source, nl]
task: Text Classification
language: nl
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `VaccinChatSentenceClassifierDutch_fromBERTje2_DAdialogQonly` is a Dutch model originally trained by `Jeska`.

## Predicted Entities

`faq_ask_taxi`, `faq_ask_twijfel_ivm_vaccinatie`, `faq_ask_naaldangst`, `faq_ask_positieve_test_na_vaccin`, `faq_ask_experimenteel`, `faq_ask_risicopatient`, `faq_ask_geen_uitnodiging`, `faq_ask_beschermingspercentage`, `faq_ask_vaccin_doorgeven`, `faq_ask_curevac`, `faq_ask_waarom`, `nlu_fallback`, `faq_ask_bijwerking_moderna`, `faq_ask_risicopatient_kanker`, `faq_ask_verschillen`, `faq_ask_keuze`, `faq_ask_huisarts`, `faq_ask_wie_doet_inenting`, `chitchat_ask_hi`, `faq_ask_algemeen_info`, `faq_ask_tijd_tot_tweede_dosis`, `faq_ask_twijfel_ontwikkeling`, `faq_ask_eerst_weigeren`, `faq_ask_hoe_weet_overheid`, `faq_ask_wanneer_iedereen_gevaccineerd`, `faq_ask_jong_en_gezond`, `faq_ask_mondmasker`, `faq_ask_privacy`, `faq_ask_derde_prik`, `faq_ask_moderna`, `faq_ask_vaccine_covid_gehad`, `faq_ask_betrouwbaar`, `faq_ask_hersenziekte`, `faq_ask_waarom_niet_verplicht`, `faq_ask_bijwerking_pfizer`, `faq_ask_buitenlander`, `chitchat_ask_bye`, `faq_ask_wie_ben_ik`, `faq_ask_quarantaine`, `faq_ask_wie_nu`, `faq_ask_beschermen`, `faq_ask_mantelzorger`, `faq_ask_testen`, `faq_ask_borstvoeding`, `faq_ask_afspraak_afzeggen`, `faq_ask_twijfel_effectiviteit`, `faq_ask_betalen_voor_vaccin`, `faq_ask_welk_vaccin_krijg_ik`, `faq_ask_vaccinatiecentrum`, `faq_ask_logistiek_veilig`, `faq_ask_aantal_gevaccineerd`, `faq_ask_tweede_dosis_vervroegen`, `faq_ask_corona_vermijden`, `faq_ask_info_vaccins`, `faq_ask_risicopatient_immuunziekte`, `faq_ask_in_vaccin`, `test`, `faq_ask_geen_risicopatient`, `faq_ask_twijfel_inhoud`, `faq_ask_keuze_vaccinatiecentrum`, `faq_ask_nadelen`, `faq_ask_astrazeneca_prik_2`, `faq_ask_twijfel_vrijheid`, `faq_ask_bijwerking_AZ`, `faq_ask_contra_ind`, `faq_ask_gestockeerd`, `faq_ask_wanneer_algemene_bevolking`, `faq_ask_wat_is_vaccin`, `faq_ask_waarom_twijfel`, `faq_ask_veelgestelde_vragen`, `faq_ask_gezondheidstoestand_gekend`, `faq_ask_risicopatient_diabetes`, `faq_ask_vrijwilliger`, `faq_ask_wat_is_corona`, `faq_ask_iedereen`, `chitchat_ask_hi_fr`, `faq_ask_nuchter`, `faq_ask_wat_na_vaccinatie`, `faq_ask_alternatieve_medicatie`, `faq_ask_bijwerking_algemeen`, `faq_ask_begeleiding`, `faq_ask_duur_vaccinatie`, `faq_ask_janssen`, `faq_ask_hoeveel_dosissen`, `faq_ask_hartspierontsteking`, `faq_ask_bijwerking_lange_termijn`, `faq_ask_dna`, `faq_ask_gif_in_vaccin`, `faq_ask_planning_eerstelijnszorg`, `faq_ask_reproductiegetal`, `chitchat_ask_thanks`, `faq_ask_problemen_uitnodiging`, `faq_ask_covid_door_vaccin`, `faq_ask_combi`, `faq_ask_tweede_dosis_afspraak`, `faq_ask_kosjer_halal`, `get_started`, `faq_ask_vrijwillig_Janssen`, `faq_ask_groepsimmuniteit`, `faq_ask_smaakverlies`, `faq_ask_astrazeneca_bloedklonters`, `faq_ask_complottheorie_Bill_Gates`, `faq_ask_ontwikkeling`, `faq_ask_vaccin_immuunsysteem`, `faq_ask_magnetisch`, `faq_ask_mrna_vs_andere_vaccins`, `faq_ask_test_voor_vaccin`, `faq_ask_betrouwbare_bronnen`, `faq_ask_astrazeneca`, `faq_ask_man_vrouw_verschillen`, `faq_ask_twijfel_bijwerkingen`, `faq_ask_eerste_prik_buitenland`, `faq_ask_sneller_aan_de_beurt`, `faq_ask_complottheorie_5G`, `faq_ask_leveringen`, `faq_ask_essentieel_beroep`, `faq_ask_geen_antwoord`, `faq_ask_twijfel_vaccins_zelf`, `faq_ask_waarom_twee_prikken`, `faq_ask_andere_vaccins`, `faq_ask_beschermingsduur`, `faq_ask_complottheorie`, `faq_ask_uit_flacon`, `faq_ask_qvax_probleem`, `faq_ask_waar_en_wanneer`, `faq_ask_onvruchtbaar`, `faq_ask_janssen_een_dosis`, `chitchat_ask_hoe_gaat_het`, `faq_ask_probleem_registratie`, `faq_ask_kinderen`, `faq_ask_trage_start`, `faq_ask_timing_andere_vaccins`, `faq_ask_uitnodiging_na_vaccinatie`, `faq_ask_snel_ontwikkeld`, `faq_ask_vakantie`, `faq_ask_foetus`, `faq_ask_risicopatient_luchtwegaandoening`, `faq_ask_bijwerking_JJ`, `faq_ask_risicopatient_hartvaat`, `faq_ask_afspraak_gemist`, `faq_ask_meer_bijwerkingen_tweede_dosis`, `faq_ask_zwanger`, `faq_ask_pijnstiller`, `faq_ask_verplicht`, `faq_ask_autisme_na_vaccinatie`, `faq_ask_chronisch_ziek`, `faq_ask_wilsonbekwaam`, `faq_ask_vaccin_variant`, `faq_ask_auto-immuun`, `faq_ask_besmetten_na_vaccin`, `faq_ask_huisdieren`, `faq_ask_prioritaire_gropen`, `faq_ask_maximaal_een_dosis`, `faq_ask_goedkeuring`, `faq_ask_wie_is_risicopatient`, `faq_ask_pfizer`, `faq_ask_bijsluiter`, `faq_ask_corona_is_griep`, `faq_ask_welke_vaccin`, `faq_ask_vaccine_covid_gehad_effect`, `faq_ask_waarom_ouderen_eerst`, `faq_ask_vegan`, `faq_ask_bloed_geven`, `faq_ask_oplopen_vaccinatie`, `faq_ask_minder_mobiel`, `faq_ask_hoe_dodelijk`, `chitchat_ask_hi_en`, `faq_ask_logistiek`, `faq_ask_attest`, `chitchat_ask_hi_de`, `faq_ask_astrazeneca_bij_ouderen`, `faq_ask_planning_ouderen`, `faq_ask_motiveren`, `faq_ask_uitnodiging_afspraak_kwijt`, `chitchat_ask_name`, `faq_ask_phishing`, `faq_ask_twijfel_praktisch`, `faq_ask_wat_is_rna`, `faq_ask_aantal_gevaccineerd_wereldwijd`, `faq_ask_allergisch_na_vaccinatie`, `faq_ask_twijfel_noodzaak`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_vaccinchatsentenceclassifierdutch_frombertje2_dadialogqonly_nl_4.1.0_3.0_1663607388718.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_vaccinchatsentenceclassifierdutch_frombertje2_dadialogqonly_nl_4.1.0_3.0_1663607388718.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_vaccinchatsentenceclassifierdutch_frombertje2_dadialogqonly","nl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["Ik hou van Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_vaccinchatsentenceclassifierdutch_frombertje2_dadialogqonly","nl") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("Ik hou van Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_vaccinchatsentenceclassifierdutch_frombertje2_dadialogqonly|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|nl|
|Size:|410.1 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/Jeska/VaccinChatSentenceClassifierDutch_fromBERTje2_DAdialogQonly