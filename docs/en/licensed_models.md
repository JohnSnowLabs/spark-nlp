---
layout: article
title: Licensed Models
permalink: /docs/en/licensed_models
key: docs-licensed-models
modify_date: "2020-02-16"
---

## Pretrained Models

Pretrained Models moved to its own dedicated repository.
Please follow this link for updated list:
[https://github.com/JohnSnowLabs/spark-nlp-models](https://github.com/JohnSnowLabs/spark-nlp-models)
{:.success}

### English

`pretrained(name, lang)` function to use


It is required to specify 3rd argument to `pretrained(name, lang, loc)` function (location) to add the location of these

| Model                    | Name                       | Build            | Notes                                                                                          | Description | location        |
|:-------------------------|:---------------------------|:-----------------|:-----------------------------------------------------------------------------------------------|:------------|:----------------|
| NerDLModel               | `ner_clinical`             | 2.0.2-2019.04.30 |                                                                                                |             | clinical/models |
| NerDLModel               | `ner_clinical_noncontrib`  | 2.3.0-2019.11.14 |                                                                                                |             | clinical/models |
| NerDLModel               | `ner_bionlp`               | 2.3.4-2019.11.27 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-bionlp-ner)         |             | clinical/models |
| NerDLModel               | `ner_bionlp_noncontrib`    | 2.3.4-2019.11.27 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-bionlp-ner)         |             | clinical/models |
| NerDLModel               | `deidentify_dl`            | 2.0.2-2019.06.04 |                                                                                                |             | clinical/models |
| AssertionDLModel         | `assertion_dl`             | 2.3.4-2019.11.27 |                                                                                                |             | clinical/models |
| AssertionLogRegModel     | `assertion_ml`             | 2.3.4-2019.11.27 |                                                                                                |             | clinical/models |
| DeIdentificationModel    | `deidentify_rb`            | 2.0.2-2019.06.04 |                                                                                                |             | clinical/models |
| WordEmbeddingsModel      | `embeddings_clinical`      | 2.0.2-2019.05.21 |                                                                                                |             | clinical/models |
| WordEmbeddingsModel      | `embeddings_icdoem`        | 2.3.2-2019.11.12 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-icd-embeddings)     |             | clinical/models |
| PerceptronModel          | `pos_clinical`             | 2.0.2-2019.04.30 |                                                                                                |             | clinical/models |
| EntityResolverModel      | `resolve_icd10`            | 2.0.2-2019.06.05 |                                                                                                |             | clinical/models |
| EntityResolverModel      | `resolve_icd10cm_cl_em`    | 2.0.8-2019.06.28 |                                                                                                |             | clinical/models |
| EntityResolverModel      | `resolve_icd10pcs_cl_em`   | 2.0.8-2019.06.28 |                                                                                                |             | clinical/models |
| EntityResolverModel      | `resolve_cpt_cl_em`        | 2.0.8-2019.06.28 |                                                                                                |             | clinical/models |
| EntityResolverModel      | `resolve_icd10cm_icdem`    | 2.2.0-2019.10.03 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-icd-embeddings)     |             | clinical/models |
| EntityResolverModel      | `resolve_icd10cm_icdoem`   | 2.3.2-2019.11.13 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-icd-embeddings)     |             | clinical/models |
| EntityResolverModel      | `resolve_cpt_icdoem`       | 2.3.2-2019.11.13 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-icd-embeddings)     |             | clinical/models |
| EntityResolverModel      | `resolve_icdo_icdoem`      | 2.3.2-2019.11.14 |                                                                                                |             | clinical/models |
| ContextSpellCheckerModel | `spellcheck_dl`            | 2.2.2-2019.11.12 |                                                                                                |             | clinical/models |
| TextMatcherModel         | `textmatch_icdo_ner_n2c4`  | 2.3.3-2019.11.22 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-icd-embeddings)     |             | clinical/models |
| TextMatcherModel         | `textmatch_cpt_token_n2c1` | 2.3.3-2019.11.25 | [link](https://github.com/JohnSnowLabs/spark-nlp-models/releases/tag/2.3.4-icd-embeddings)     |             | clinical/models |
| DisambiguatorModel       | `people_disambiguator`     | 2.3.4-2019.11.27 |                                                                                                |             | clinical/models |
| ChunkEntityResolverModel | `chunkresolve_icdo_icdoem` | 2.3.3-2019.11.25 |                                                                                                |             | clinical/models |
| ChunkEntityResolverModel | `chunkresolve_cpt_icdoem`  | 2.3.3-2019.11.25 |                                                                                                |             | clinical/models |
