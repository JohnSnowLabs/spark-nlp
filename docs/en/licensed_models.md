---
layout: docs
header: true
title: Licensed Models
permalink: /docs/en/licensed_models
key: docs-licensed-models
modify_date: "2020-04-22"
---

<div class="h3-box" markdown="1">

## Pretrained Models

We are currently in the process of moving the pretrained models and pipelines to a Model Hub that you can explore here: 
[Models Hub](/models)
{:.success}

</div>

### English

It is required to specify 3rd argument to `pretrained(name, lang, location)` function to add the location of these

{:.table-model-big}
| Model                      | Name                                            | Build   | Download                                                                                                                                                                                      |
|----------------------------|-------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  AssertionDLModel          |  assertion_dl_large                             |  2.5.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_large_en_2.5.0_2.4_1590022282256.zip 'Download')                            |
|  AssertionDLModel          |  assertion_dl                                   |  2.4.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_en_2.4.0_2.4_1580237286004.zip 'Download')                                  |
|  AssertionLogRegModel      |  assertion_ml                                   |  2.4.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/assertion_ml_en_2.4.0_2.4_1580237286004.zip 'Download')                                  |
|  ChunkEntityResolverModel  |  chunkresolve_cpt_clinical                      |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_cpt_clinical_en_2.4.5_2.4_1587491373378.zip 'Download')                     |
|  ChunkEntityResolverModel  |  chunkresolve_icd10cm_clinical                  |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_clinical_en_2.4.5_2.4_1587491222166.zip 'Download')                 |
|  ChunkEntityResolverModel  |  chunkresolve_icd10cm_diseases_clinical         |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_diseases_clinical_en_2.4.5_2.4_1588105984876.zip 'Download')        |
|  ChunkEntityResolverModel  |  chunkresolve_icd10cm_injuries_clinical         |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_injuries_clinical_en_2.4.5_2.4_1588103825347.zip 'Download')        |
|  ChunkEntityResolverModel  |  chunkresolve_icd10cm_musculoskeletal_clinical  |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_musculoskeletal_clinical_en_2.4.5_2.4_1588103998999.zip 'Download') |
|  ChunkEntityResolverModel  |  chunkresolve_icd10cm_neoplasms_clinical        |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_neoplasms_clinical_en_2.4.5_2.4_1588108205630.zip 'Download')       |
|  ChunkEntityResolverModel  |  chunkresolve_icd10cm_puerile_clinical          |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_puerile_clinical_en_2.4.5_2.4_1588103916781.zip 'Download')         |
|  ChunkEntityResolverModel  |  chunkresolve_icd10pcs_clinical                 |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10pcs_clinical_en_2.4.5_2.4_1587491320087.zip 'Download')                |
|  ChunkEntityResolverModel  |  chunkresolve_icdo_clinical                     |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icdo_clinical_en_2.4.5_2.4_1587491354644.zip 'Download')                    |
|  ChunkEntityResolverModel  |  chunkresolve_loinc_clinical                    |  2.5.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_loinc_clinical_en_2.5.0_2.4_1589599195201.zip 'Download')                   |
|  ChunkEntityResolverModel  |  chunkresolve_rxnorm_cd_clinical                |  2.5.1  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_cd_clinical_en_2.5.1_2.4_1595813950836.zip 'Download')               |
|  ChunkEntityResolverModel  |  chunkresolve_rxnorm_sbd_clinical               |  2.5.1  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_sbd_clinical_en_2.5.1_2.4_1595813912622.zip 'Download')              |
|  ChunkEntityResolverModel  |  chunkresolve_rxnorm_scd_clinical               |  2.5.1  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_rxnorm_scd_clinical_en_2.5.1_2.4_1595813884363.zip 'Download')              |
|  ChunkEntityResolverModel  |  chunkresolve_snomed_findings_clinical          |  2.5.1  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_snomed_findings_clinical_en_2.5.1_2.4_1592617161564.zip 'Download')         |
|  ContextSpellCheckerModel  |  spellcheck_clinical                            |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_2.4.2_2.4_1587146727460.zip 'Download')                           |
|  DeIdentificationModel     |  deidentify_rb_no_regex                         |  2.5.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/deidentify_rb_no_regex_en_2.5.0_2.4_1589924063833.zip 'Download')                        |
|  DeIdentificationModel     |  deidentify_rb                                  |  2.0.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/deidentify_rb_en_2.0.2_2.4_1559672122511.zip 'Download')                                 |
|  DeIdentificatoinModel     |  deidentify_large                               |  2.5.1  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/deidentify_large_en_2.5.1_2.4_1595199111307.zip 'Download')                              |
|  NerDLModel                |  ner_anatomy                                    |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_en_2.4.2_2.4_1587513307751.zip 'Download')                                   |
|  NerDLModel                |  ner_bionlp                                     |  2.4.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_en_2.4.0_2.4_1580237286004.zip 'Download')                                    |
|  NerDLModel                |  ner_cellular                                   |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_cellular_en_2.4.2_2.4_1587513308751.zip 'Download')                                  |
|  NerDLModel                |  ner_clinical_large                             |  2.5.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_large_en_2.5.0_2.4_1590021302624.zip 'Download')                            |
|  NerDLModel                |  ner_clinical                                   |  2.4.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_en_2.4.0_2.4_1580237286004.zip 'Download')                                  |
|  NerDLModel                |  ner_deid_enriched                              |  2.5.3  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_en_2.5.3_2.4_1594170530497.zip 'Download')                             |
|  NerDLModel                |  ner_deid_large                                 |  2.5.3  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_deid_large_en_2.5.3_2.4_1595427435246.zip 'Download')                                |
|  NerDLModel                |  ner_diseases                                   |  2.4.4  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_diseases_en_2.4.4_2.4_1584452534235.zip 'Download')                                  |
|  NerDLModel                |  ner_drugs                                      |  2.4.4  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_en_2.4.4_2.4_1584452534235.zip 'Download')                                     |
|  NerDLModel                |  ner_events_clinical                            |  2.5.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_events_clinical_en_2.5.0_2.4_1590021303624.zip 'Download')                           |
|  NerDLModel                |  ner_healthcare                                 |  2.4.4  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_en_2.4.4_2.4_1585188313964.zip 'Download')                                |
|  NerDLModel                |  ner_jsl_enriched                               |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_en_2.4.2_2.4_1587513303751.zip 'Download')                              |
|  NerDLModel                |  ner_jsl                                        |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_en_2.4.2_2.4_1587513304751.zip 'Download')                                       |
|  NerDLModel                |  ner_medmentions_coarse                         |  2.5.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_medmentions_coarse_en_2.5.0_2.4_1590265407598.zip 'Download')                        |
|  NerDLModel                |  ner_posology_large                             |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_posology_large_en_2.4.2_2.4_1587513302751.zip 'Download')                            |
|  NerDLModel                |  ner_posology_small                             |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_posology_small_en_2.4.2_2.4_1587513301751.zip 'Download')                            |
|  NerDLModel                |  ner_posology                                   |  2.4.4  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_posology_en_2.4.4_2.4_1584452534235.zip 'Download')                                  |
|  NerDLModel                |  ner_risk_factors                               |  2.4.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_risk_factors_en_2.4.2_2.4_1587513300751.zip 'Download')                              |
|  PerceptronModel           |  pos_clinical                                   |  2.0.2  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/pos_clinical_en_2.0.2_2.4_1556660550177.zip 'Download')                                  |
|  RelationExtractionModel   |  re_clinical                                    |  2.5.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/re_clinical_en_2.5.5_2.4_1596928426753.zip 'Download')                                   |
|  TextMatcherModel          |  textmatch_cpt_token                            |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/textmatch_cpt_token_en_2.4.5_2.4_1587495106014.zip 'Download')                           |
|  TextMatcherModel          |  textmatch_icdo_ner                             |  2.4.5  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/textmatch_icdo_ner_en_2.4.5_2.4_1587495006987.zip 'Download')                            |
|  WordEmbeddingsModel       |  embeddings_clinical                            |  2.4.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/embeddings_clinical_en_2.4.0_2.4_1580237286004.zip 'Download')                           |
|  WordEmbeddingsModel       |  embeddings_healthcare_100d                     |  2.5.0  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/embeddings_healthcare_100d_en_2.5.0_2.4_1590794626292.zip 'Download')                    |
|  WordEmbeddingsModel       |  embeddings_healthcare                          |  2.4.4  | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/embeddings_healthcare_en_2.4.4_2.4_1585188313964.zip 'Download')                         |
