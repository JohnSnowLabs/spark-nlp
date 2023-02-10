---
layout: model
title: Extract the Names of Drugs & Chemicals
author: John Snow Labs
name: ner_chemd_clinical
date: 2021-11-04
tags: [chemdner, chemd, ner, clinical, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.0
spark_version: 2.4
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained with `clinical_embeddings` to extract the names of chemical compounds and drugs in medical texts. The entities that can be detected are as follows :

- `SYSTEMATIC` : Systematic names of chemical mentions, e.g. IUPAC and IUPAC-like names.(e.g. 2-Acetoxybenzoic acid, 2-Acetoxybenzenecarboxylic acid)
- `IDENTIFIERS` : Database identifiers of chemicals: CAS numbers, PubChem identifiers, registry numbers and ChEBI and CHEMBL ids. (e.g. CAS Registry Number: 501-36-0445154 CID 445154, CHEBI:28262, CHEMBL504)
- `FORMULA` : Mentions of molecular formula, SMILES, InChI, InChIKey. (e.g. CC(=O)Oc1ccccc1C(=O)O, C9H8O4)
- `TRIVIAL` : Trivial, trade (brand), common or generic names of compounds. It includes International Nonproprietary Name (INN) as well as British Approved Name (BAN) and United States Adopted Name (USAN). (e.g. Aspirin, Acylpyrin, paracetamol)
- `ABBREVIATION` : Mentions of abbreviations and acronyms of chemicals compounds and drugs. (e.g. DMSO, GABA)
- `FAMILY`: Chemical families that can be associated to some chemical structure are also included. (e.g. Iodopyridazines (FAMILY- SYSTEMATIC))
- `MULTIPLE` : Mentions that do correspond to chemicals that are not described in a continuous string of characters. This is often the case of mentions of multiple chemicals joined by coordinated clauses. (e.g. thieno2,3-d and thieno3,2-d fused oxazin-4-ones)

## Predicted Entities

`SYSTEMATIC`, `IDENTIFIERS`, `FORMULA`, `TRIVIAL`, `ABBREVIATION`, `FAMILY`, `MULTIPLE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CHEMD/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chemd_clinical_en_3.3.0_2.4_1636027285679.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_chemd_clinical_en_3.3.0_2.4_1636027285679.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

chemd_ner = MedicalNerModel.pretrained('ner_chemd_clinical', 'en', 'clinical/models') \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, chemd_ner, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

chemd_model = nlpPipeline.fit(empty_data)

results = chemd_model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""Isolation, Structure Elucidation, and Iron-Binding Properties of Lystabactins, Siderophores Isolated from a Marine Pseudoalteromonas sp. The marine bacterium Pseudoalteromonas sp. S2B, isolated from the Gulf of Mexico after the Deepwater Horizon oil spill, was found to produce lystabactins A, B, and C (1-3), three new siderophores. The structures were elucidated through mass spectrometry, amino acid analysis, and NMR. The lystabactins are composed of serine (Ser), asparagine (Asn), two formylated/hydroxylated ornithines (FOHOrn), dihydroxy benzoic acid (Dhb), and a very unusual nonproteinogenic amino acid, 4,8-diamino-3-hydroxyoctanoic acid (LySta). The iron-binding properties of the compounds were investigated through a spectrophotometric competition."""]})))
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val chemd_ner = MedicalNerModel.pretrained("ner_chemd_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, chemd_ner, ner_converter))

val result = pipeline.fit(Seq.empty["Isolation, Structure Elucidation, and Iron-Binding Properties of Lystabactins, Siderophores Isolated from a Marine Pseudoalteromonas sp. The marine bacterium Pseudoalteromonas sp. S2B, isolated from the Gulf of Mexico after the Deepwater Horizon oil spill, was found to produce lystabactins A, B, and C (1-3), three new siderophores. The structures were elucidated through mass spectrometry, amino acid analysis, and NMR. The lystabactins are composed of serine (Ser), asparagine (Asn), two formylated/hydroxylated ornithines (FOHOrn), dihydroxy benzoic acid (Dhb), and a very unusual nonproteinogenic amino acid, 4,8-diamino-3-hydroxyoctanoic acid (LySta). The iron-binding properties of the compounds were investigated through a spectrophotometric competition."].toDS.toDF("text")).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.chemd").predict("""Isolation, Structure Elucidation, and Iron-Binding Properties of Lystabactins, Siderophores Isolated from a Marine Pseudoalteromonas sp. The marine bacterium Pseudoalteromonas sp. S2B, isolated from the Gulf of Mexico after the Deepwater Horizon oil spill, was found to produce lystabactins A, B, and C (1-3), three new siderophores. The structures were elucidated through mass spectrometry, amino acid analysis, and NMR. The lystabactins are composed of serine (Ser), asparagine (Asn), two formylated/hydroxylated ornithines (FOHOrn), dihydroxy benzoic acid (Dhb), and a very unusual nonproteinogenic amino acid, 4,8-diamino-3-hydroxyoctanoic acid (LySta). The iron-binding properties of the compounds were investigated through a spectrophotometric competition.""")
```

</div>

## Results

```bash
+----------------------------------+------------+
|chunk                             |ner_label   |
+----------------------------------+------------+
|Lystabactins                      |FAMILY      |
|lystabactins A, B, and C          |MULTIPLE    |
|amino acid                        |FAMILY      |
|lystabactins                      |FAMILY      |
|serine                            |TRIVIAL     |
|Ser                               |FORMULA     |
|asparagine                        |TRIVIAL     |
|Asn                               |FORMULA     |
|formylated/hydroxylated ornithines|FAMILY      |
|FOHOrn                            |FORMULA     |
|dihydroxy benzoic acid            |SYSTEMATIC  |
|amino acid                        |FAMILY      |
|4,8-diamino-3-hydroxyoctanoic acid|SYSTEMATIC  |
|LySta                             |ABBREVIATION|
+----------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chemd_clinical|
|Compatibility:|Healthcare NLP 3.3.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

CHEMDNER Corpus. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4331685/

## Benchmarking

```bash
+------------+------+-----+-----+------+---------+------+------+
|      entity|    tp|   fp|   fn| total|precision|recall|    f1|
+------------+------+-----+-----+------+---------+------+------+
|     FORMULA|2296.0|196.0| 99.0|2395.0|   0.9213|0.9587|0.9396|
|  IDENTIFIER| 294.0| 33.0| 25.0| 319.0|   0.8991|0.9216|0.9102|
|    MULTIPLE| 310.0|284.0| 58.0| 368.0|   0.5219|0.8424|0.6445|
|      FAMILY|1865.0|277.0|347.0|2212.0|   0.8707|0.8431|0.8567|
|ABBREVIATION|1648.0|188.0|158.0|1806.0|   0.8976|0.9125| 0.905|
|  SYSTEMATIC|3381.0|336.0|307.0|3688.0|   0.9096|0.9168|0.9132|
|     TRIVIAL|3862.0|255.0|241.0|4103.0|   0.9381|0.9413|0.9397|
+------------+------+-----+-----+------+---------+------+------+

+------------------+
|             macro|
+------------------+
|0.8726928630563308|
+------------------+

+------------------+
|             micro|
+------------------+
|0.9086394322529923|
+------------------+
```
