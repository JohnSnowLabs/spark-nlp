---
layout: model
title: Legal Deidentification Pipeline
author: John Snow Labs
name: legpipe_deid
date: 2023-02-22
tags: [deid, deidentification, anonymization, en, licensed]
task: [De-identification, Pipeline Legal]
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Pretrained Pipeline aimed to deidentify legal and financial documents to be compliant with data privacy regulations as GDPR and CCPA. Since the models used in this pipeline are statistical, make sure you use this model in a human-in-the-loop process to guarantee a 100% accuracy.

You can carry out both masking and obfuscation with this pipeline, on the following entities: 
`ALIAS`, `EMAIL`, `PHONE`, `PROFESSION`, `ORG`, `DATE`, `PERSON`, `ADDRESS`, `STREET`, `CITY`, `STATE`, `ZIP`, `COUNTRY`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/DEID_LEGAL/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legpipe_deid_en_1.0.0_3.0_1677077428423.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legpipe_deid_en_1.0.0_3.0_1677077428423.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("legpipe_deid", "en", "legal/models")

sample_2 = """Pizza Fusion Holdings, Inc. Franchise Agreement This Franchise Agreement (the "Agreement") is entered into as of the Agreement Date shown on the cover page between Pizza Fusion Holding, Inc., a Florida corporation, and the individual or legal entity identified on the cover page.

Source: PF HOSPITALITY GROUP INC., 9/23/2015


1. RIGHTS GRANTED 1.1. Grant of Franchise. 1.1.1 We grant you the right, and you accept the obligation, to use the Proprietary Marks and the System to operate one Restaurant (the "Franchised Business") at the Premises, in accordance with the terms of this Agreement. 

Source: PF HOSPITALITY GROUP INC., 9/23/2015


1.3. Our Limitations and Our Reserved Rights. The rights granted to you under this Agreement are not exclusive.sed Business.

Source: PF HOSPITALITY GROUP INC., 9/23/2015

"""

result = deid_pipeline.annotate(sample_2)
print("\nMasked with entity labels")
print("-"*30)
print("\n".join(result['deidentified']))
print("\nMasked with chars")
print("-"*30)
print("\n".join(result['masked_with_chars']))
print("\nMasked with fixed length chars")
print("-"*30)
print("\n".join(result['masked_fixed_length_chars']))
print("\nObfuscated")
print("-"*30)
print("\n".join(result['obfuscated']))
```

</div>

## Results

```bash
Masked with entity labels
------------------------------
<PARTY>. <DOC> This <DOC> (the <ALIAS>) is entered into as of the Agreement Date shown on the cover page between <PARTY> a Florida corporation, and the individual or legal entity identified on the cover page.
Source: <PARTY>., <DATE>


1.
<PARTY> 1.1.
<PARTY>.
1.1.1 We grant you the right, and you accept the obligation, to use the <PARTY> and the System to operate one Restaurant (the <ALIAS>) at the Premises, in accordance with the terms of this Agreement.
Source: <PARTY>., <DATE>


1.3.
Our <PARTY> and <PARTY>.
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: <PARTY>., <DATE>

Masked with chars
------------------------------
[************************]. [*****************] This [*****************] (the [*********]) is entered into as of the Agreement Date shown on the cover page between [*************************] a Florida corporation, and the individual or legal entity identified on the cover page.
Source: [**********************]., [*******]


1.
[************] 1.1.
[****************].
1.1.1 We grant you the right, and you accept the obligation, to use the [***************] and the System to operate one Restaurant (the [*******************]) at the Premises, in accordance with the terms of this Agreement.
Source: [**********************]., [*******]


1.3.
Our [*********] and [*****************].
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: [**********************]., [*******]

Masked with fixed length chars
------------------------------
****. **** This **** (the ****) is entered into as of the Agreement Date shown on the cover page between **** a Florida corporation, and the individual or legal entity identified on the cover page.
Source: ****., ****


1.
**** 1.1.
****.
1.1.1 We grant you the right, and you accept the obligation, to use the **** and the System to operate one Restaurant (the ****) at the Premises, in accordance with the terms of this Agreement.
Source: ****., ****


1.3.
Our **** and ****.
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: ****., ****

Obfuscated
------------------------------
SESA CO.. Estate Document This Estate Document (the (the "Contract")) is entered into as of the Agreement Date shown on the cover page between Clarus llc. a Florida corporation, and the individual or legal entity identified on the cover page.
Source: SESA CO.., 11/7/2016


1.
SESA CO. 1.1.
Clarus llc..
1.1.1 We grant you the right, and you accept the obligation, to use the John Snow Labs Inc and the System to operate one Restaurant (the (the" Agreement")) at the Premises, in accordance with the terms of this Agreement.
Source: SESA CO.., 11/7/2016


1.3.
Our MGT Trust Company, LLC. and John Snow Labs Inc.
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: SESA CO.., 11/7/2016
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legpipe_deid|
|Type:|pipeline|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|965.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- RoBertaEmbeddings
- LegalNerModel
- NerConverterInternalModel
- LegalNerModel
- NerConverter
- ZeroShotNerModel
- NerConverterInternalModel
- ContextualParserModel
- ContextualParserModel
- ContextualParserModel
- ChunkMergeModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
- DeIdentificationModel
