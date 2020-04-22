---
layout: article
title: Licensed Models
permalink: /docs/en/licensed_models
key: docs-licensed-models
modify_date: "2020-04-22"
---

## Pretrained Models

Pretrained Models moved to its own dedicated repository.
Please follow this link for updated list:
[GitHub Repository for Pretrained Models](https://github.com/JohnSnowLabs/spark-nlp-models#licensed-enterprise)
{:.success}

### English

It is required to specify 3rd argument to `pretrained(name, lang, location)` function to add the location of these

| Model                         | Name                                | Build   | Download                                                                                                                                                                          |
|-------------------------------|-------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `AssertionDLModel`            | `assertion_dl`                      | `2.4.0` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_en_2.4.0_2.4_1580237286004.zip 'Download')                      |
| `AssertionLogRegModel`        | `assertion_ml`                      | `2.4.0` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/assertion_ml_en_2.4.0_2.4_1580237286004.zip 'Download')                      |
| `BertEmbeddings`              | `biobert_clinical_cased`            | `2.3.1` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/biobert_clinical_cased_en_2.3.1_2.4_1574522054965.zip 'Download')            |
| `BertEmbeddings`              | `biobert_discharge_cased`           | `2.3.1` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/biobert_discharge_cased_en_2.3.1_2.4_1574522388638.zip 'Download')           |
| `BertEmbeddings`              | `biobert_pmc_cased`                 | `2.3.1` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/biobert_pmc_cased_en_2.3.1_2.4_1574521384805.zip 'Download')                 |
| `BertEmbeddings`              | `biobert_pubmed_cased`              | `2.3.1` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/biobert_pubmed_cased_en_2.3.1_2.4_1574521132506.zip 'Download')              |
| `BertEmbeddings`              | `biobert_pubmed_pmc_cased`          | `2.3.1` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/biobert_pubmed_pmc_cased_en_2.3.1_2.4_1574521728558.zip 'Download')          |
| `ChunkEntityResolverModel`    | `chunkresolve_icdo_clinical`        | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icdo_clinical_en_2.4.5_2.4_1587491354644.zip 'Download')        |
| `ChunkEntityResolverModel`    | `chunkresolve_icd10pcs_clinical`    | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10pcs_clinical_en_2.4.5_2.4_1587491320087.zip 'Download')    |
| `ChunkEntityResolverModel`    | `chunkresolve_icd10cm_clinical`     | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_icd10cm_clinical_en_2.4.5_2.4_1587491222166.zip 'Download')     |
| `ChunkEntityResolverModel`    | `chunkresolve_cpt_clinical`         | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/chunkresolve_cpt_clinical_en_2.4.5_2.4_1587491373378.zip 'Download')         |
| `ContextSpellCheckerModel`    | `context_spell_med`                 | `2.0.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/context_spell_med_en_2.0.2_2.4_1564584130634.zip 'Download')                 |
| `ContextSpellCheckerModel`    | `spellcheck_dl`                     | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/spellcheck_dl_en_2.4.2_2.4_1587056595200.zip 'Download')                     |
| `ContextSpellCheckerModel`    | `spellcheck_clinical`               | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/spellcheck_clinical_en_2.4.2_2.4_1587146727460.zip 'Download')               |
| `DeIdentificationModel`       | `deidentify_rb`                     | `2.0.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/deidentify_rb_en_2.0.2_2.4_1559672122511.zip 'Download')                     |
| `EnsembleEntityResolverModel` | `ensembleresolve_snomed_clinical`   | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ensembleresolve_snomed_clinical_en_2.4.5_2.4_1587296548545.zip 'Download')   |
| `EnsembleEntityResolverModel` | `ensembleresolve_rxnorm_healthcare` | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ensembleresolve_rxnorm_healthcare_en_2.4.5_2.4_1587302681254.zip 'Download') |
| `EnsembleEntityResolverModel` | `ensembleresolve_snomed_healthcare` | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ensembleresolve_snomed_healthcare_en_2.4.5_2.4_1587298549235.zip 'Download') |
| `EnsembleEntityResolverModel` | `ensembleresolve_rxnorm_clinical`   | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ensembleresolve_rxnorm_clinical_en_2.4.5_2.4_1587300549721.zip 'Download')   |
| `NerDLModel`                  | `ner_healthcare`                    | `2.4.4` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_en_2.4.4_2.4_1585188313964.zip 'Download')                    |
| `NerDLModel`                  | `ner_posology_small`                | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_posology_small_en_2.4.2_2.4_1587513301751.zip 'Download')                |
| `NerDLModel`                  | `ner_posology_large`                | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_posology_large_en_2.4.2_2.4_1587513302751.zip 'Download')                |
| `NerDLModel`                  | `ner_posology`                      | `2.4.4` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_posology_en_2.4.4_2.4_1584452534235.zip 'Download')                      |
| `NerDLModel`                  | `ner_jsl_enriched`                  | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_en_2.4.2_2.4_1587513303751.zip 'Download')                  |
| `NerDLModel`                  | `ner_jsl`                           | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_en_2.4.2_2.4_1587513304751.zip 'Download')                           |
| `NerDLModel`                  | `ner_drugs`                         | `2.4.4` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_en_2.4.4_2.4_1584452534235.zip 'Download')                         |
| `NerDLModel`                  | `ner_diseases`                      | `2.4.4` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_diseases_en_2.4.4_2.4_1584452534235.zip 'Download')                      |
| `NerDLModel`                  | `ner_deid_large`                    | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_deid_large_en_2.4.2_2.4_1587513305751.zip 'Download')                    |
| `NerDLModel`                  | `ner_deid_enriched`                 | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_en_2.4.2_2.4_1587513306751.zip 'Download')                 |
| `NerDLModel`                  | `ner_clinical`                      | `2.4.0` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_en_2.4.0_2.4_1580237286004.zip 'Download')                      |
| `NerDLModel`                  | `ner_cellular`                      | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_cellular_en_2.4.2_2.4_1587513308751.zip 'Download')                      |
| `NerDLModel`                  | `ner_bionlp`                        | `2.4.0` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_en_2.4.0_2.4_1580237286004.zip 'Download')                        |
| `NerDLModel`                  | `ner_anatomy`                       | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_en_2.4.2_2.4_1587513307751.zip 'Download')                       |
| `NerDLModel`                  | `deidentify_dl`                     | `2.4.0` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/deidentify_dl_en_2.4.0_2.4_1580237286004.zip 'Download')                     |
| `NerDLModel`                  | `ner_risk_factors`                  | `2.4.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/ner_risk_factors_en_2.4.2_2.4_1587513300751.zip 'Download')                  |
| `PerceptronModel`             | `pos_clinical`                      | `2.0.2` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/pos_clinical_en_2.0.2_2.4_1556660550177.zip 'Download')                      |
| `TextMatcherModel`            | `textmatch_icdo_ner`                | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/textmatch_icdo_ner_en_2.4.5_2.4_1587495006987.zip 'Download')                |
| `TextMatcherModel`            | `textmatch_cpt_token`               | `2.4.5` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/textmatch_cpt_token_en_2.4.5_2.4_1587495106014.zip 'Download')               |
| `WordEmbeddingsModel`         | `embeddings_clinical`               | `2.4.0` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/embeddings_clinical_en_2.4.0_2.4_1580237286004.zip 'Download')               |
| `WordEmbeddingsModel`         | `embeddings_healthcare`             | `2.4.4` | [:floppy_disk:](https://s3.console.aws.amazon.com/s3/object/auxdata.johnsnowlabs.com/clinical/models/embeddings_healthcare_en_2.4.4_2.4_1585188313964.zip 'Download')             |
