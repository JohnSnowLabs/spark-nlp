---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.5.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_5_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.5.1
We are glad to announce that 3.5.1 version of Spark NLP for Healthcare has been released!

#### Highlights
- **Deidentification**:
  - New **Portuguese** **Deidentification** NER models and pretrained pipeline. This is the 6th supported language for deidentification (English, German, Spanish, Italian, French and Portuguese).
- **New pretrained models and pipelines**:
  - New **RxNorm** Sentence Entity Resolver model to map and extract pharmaceutical actions (e.g. analgesic, hypoglycemic) as well as treatments (e.g. backache, diabetes) along with the RxNorm code resolved (`sbiobertresolve_rxnorm_action_treatment`)
  - New **RCT** classification models and pretrained pipelines to classify the sections within the abstracts of scientific articles regarding randomized clinical trials (RCT). (`rct_binary_classifier_use`, `rct_binary_classifier_biobert`, `bert_sequence_classifier_binary_rct_biobert`, `rct_binary_classifier_use_pipeline`, `rct_binary_classifier_biobert_pipeline`, `bert_sequence_classifier_binary_rct_biobert_pipeline`)
- **New features**:
  - Add `getClasses()` attribute for `MedicalBertForTokenClassifier` and `MedicalBertForSequenceClassification` to find out the entity classes of the models
  - Download the AnnotatorModels from the healthcare library using the Healthcare version instead of the open source version (the pretrained models were used to be dependent on open source Spark NLP version before)
  - New functionality to download and extract clinical models from S3 via direct zip url.
- **Core improvements**:
  - Fixing the confidence scores in `MedicalNerModel` when `setIncludeAllConfidenceScores` is true
  - Graph_builder `relation_extraction` model file name extension problem with `auto` parameter.

- **List of recently updated or added models**

#### Portuguese Deidentification Models

This is the 6th supported language for deidentification (English, German, Spanish, Italian, French and Portuguese). This version includes two Portuguese deidentification models to mask or obfuscate Protected Health Information in the Portuguese language. The models are the following:

- `ner_deid_generic`:  extracts `Name`, `Profession`, `Age`, `Date`, `Contact` (Telephone numbers, Email addresses), `Location` (Address, City, Postal code, Hospital Name, Organization), `ID` (Social Security numbers, Medical record numbers) and `Sex` entities.

   See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/04/13/ner_deid_generic_pt_3_0.html) for details.

- `ner_deid_subentity`: `Patient` (name), `Hospital` (name), `Date`, `Organization`, `City`, `ID`, `Street`, `Sex`, `Email`, `ZIP`, `Profession`, `Phone`, `Country`, `Doctor` (name) and `Age`

  See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/04/13/ner_deid_subentity_pt_3_0.html) for details.

You will use the `w2v_cc_300d` Portuguese Embeddings with these models. The pipeline should look as follows:
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "pt")\
    .setInputCols(["sentence","token"])\
    .setOutputCol("embeddings")

ner_subentity = MedicalNerModel.pretrained("ner_deid_subentity", "pt", "clinical/models")\    
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner_deid_subentity")

ner_converter_subentity = NerConverter()\
    .setInputCols(["sentence","token","ner_deid_subentity"])\
    .setOutputCol("ner_chunk_subentity")

ner_generic = MedicalNerModel.pretrained("ner_deid_generic", "pt", "clinical/models")\    
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner_deid_generic")

ner_converter_generic = NerConverter()\
    .setInputCols(["sentence","token","ner_deid_generic"])\
    .setOutputCol("ner_chunk_generic")

nlpPipeline = Pipeline(stages=[
      documentAssembler,
      sentencerDL,
      tokenizer,
      word_embeddings,
      ner_subentity,
      ner_converter_subentity,
      ner_generic,
      ner_converter_generic,
      ])

text = """Detalhes do paciente.
Nome do paciente:  Pedro Gonçalves
NHC: 2569870.
Endereço: Rua Das Flores 23.
Código Postal: 21754-987.
Dados de cuidados.
Data de nascimento: 10/10/1963.
Idade: 53 anos
Data de admissão: 17/06/2016.
Doutora: Maria Santos"""

data = spark.createDataFrame([[text]]).toDF("text")
results = nlpPipeline.fit(data).transform(data)

```

Results:
```
+-----------------+-------------------------------------+
|chunk            |ner_generic_label|ner_subentity_label|
+-----------------+-------------------------------------+
|Pedro Gonçalves  |      NAME       |      PATIENT      |
|2569870          |      ID         |      ID           |
|Rua Das Flores 23|      LOCATION   |      STREET       |
|21754-987        |      LOCATION   |      ZIP          |
|10/10/1963       |      DATE       |      DATE         |
|53               |      AGE        |      AGE          |
|17/06/2016       |      DATE       |      DATE         |
|Maria Santos     |      NAME       |      DOCTOR       |
+-----------------+-------------------------------------+
```

We also include a Clinical Deidentification Pipeline for Portuguese that uses `ner_deid_subentity` NER model and also several `ContextualParsers` for rule based contextual Named Entity Recognition tasks. It's available to be used as follows:

```python
from sparknlp.pretrained import PretrainedPipeline

deid_pipeline = PretrainedPipeline("clinical_deidentification", "pt", "clinical/models")
```

The pretrained pipeline comes with Deidentification and Obfuscation capabilities as shows the following example:

```
text = """RELAÇÃO HOSPITALAR
NOME: Pedro Gonçalves
NHC: MVANSK92F09W408A
ENDEREÇO: Rua Burcardo 7
CÓDIGO POSTAL: 80139
DATA DE NASCIMENTO: 03/03/1946
IDADE: 70 anos
SEXO: Homens
E-MAIL: pgon21@tim.pt
DATA DE ADMISSÃO: 12/12/2016
DOUTORA: Eva Andrade
RELATO CLÍNICO: 70 anos, aposentado, sem alergia a medicamentos conhecida, com a seguinte história: ex-acidente de trabalho com fratura de vértebras e costelas; operado de doença de Dupuytren na mão direita e ponte ílio-femoral esquerda; diabetes tipo II, hipercolesterolemia e hiperuricemia; alcoolismo ativo, fuma 20 cigarros/dia.
Ele foi encaminhado a nós por apresentar hematúria macroscópica pós-evacuação em uma ocasião e microhematúria persistente posteriormente, com evacuação normal.
O exame físico mostrou bom estado geral, com abdome e genitais normais; o toque retal foi compatível com adenoma de próstata grau I/IV.
A urinálise mostrou 4 hemácias/campo e 0-5 leucócitos/campo; o resto do sedimento era normal.
O hemograma é normal; a bioquímica mostrou uma glicemia de 169 mg/dl e triglicerídeos 456 mg/dl; função hepática e renal são normais. PSA de 1,16 ng/ml.

DIRIGIDA A: Dr. Eva Andrade - Centro Hospitalar do Medio Ave - Avenida Dos Aliados, 56
E-MAIL: evandrade@poste.pt
"""

result = deid_pipeline.annotate(text)
```

Results:
```
|    | Sentence                       | Masked                     | Masked with Chars              | Masked with Fixed Chars   | Obfuscated                        |
|---:|:-------------------------------|:---------------------------|:-------------------------------|:--------------------------|:----------------------------------|
|  0 | RELAÇÃO HOSPITALAR             | RELAÇÃO HOSPITALAR         | RELAÇÃO HOSPITALAR             | RELAÇÃO HOSPITALAR        | RELAÇÃO HOSPITALAR                |
|    | NOME: Pedro Gonçalves          | NOME: <DOCTOR>             | NOME: [*************]          | NOME: ****                | NOME: Isabel Magalhães            |
|  1 | NHC: MVANSK92F09W408A          | NHC: <ID>                  | NHC: [**************]          | NHC: ****                 | NHC: 124 445 311                  |
|  2 | ENDEREÇO: Rua Burcardo 7       | ENDEREÇO: <STREET>         | ENDEREÇO: [************]       | ENDEREÇO: ****            | ENDEREÇO: Rua de Santa María, 100 |
|  3 | CÓDIGO POSTAL: 80139           | CÓDIGO POSTAL: <ZIP>       | CÓDIGO POSTAL: [***]           | CÓDIGO POSTAL: ****       | CÓDIGO POSTAL: 1000-306           |
|    | DATA DE NASCIMENTO: 03/03/1946 | DATA DE NASCIMENTO: <DATE> | DATA DE NASCIMENTO: [********] | DATA DE NASCIMENTO: ****  | DATA DE NASCIMENTO: 04/04/1946    |
|  4 | IDADE: 70 anos                 | IDADE: <AGE> anos          | IDADE: ** anos                 | IDADE: **** anos          | IDADE: 46 anos                    |
|  5 | SEXO: Homens                   | SEXO: <SEX>                | SEXO: [****]                   | SEXO: ****                | SEXO: Mulher                      |
|  6 | E-MAIL: pgon21@tim.pt          | E-MAIL: <EMAIL>            | E-MAIL: [***********]          | E-MAIL: ****              | E-MAIL: eric.shannon@geegle.com   |
|    | DATA DE ADMISSÃO: 12/12/2016   | DATA DE ADMISSÃO: <DATE>   | DATA DE ADMISSÃO: [********]   | DATA DE ADMISSÃO: ****    | DATA DE ADMISSÃO: 23/12/2016      |
|  7 | DOUTORA: Eva Andrade           | DOUTORA: <DOCTOR>          | DOUTORA: [*********]           | DOUTORA: ****             | DOUTORA: Isabel Magalhães         |
```

 See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/04/14/clinical_deidentification_pt_3_0.html) for details.


Check Spark NLP Portuguese capabilities in [4.7.Clinical_Deidentification_in_Portuguese.ipynb notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.7.Clinical_Deidentification_in_Portuguese.ipynb) we have prepared for you.

#### New RxNorm Sentence Entity Resolver Model (`sbiobertresolve_rxnorm_action_treatment`)

We are releasing `sbiobertresolve_rxnorm_action_treatment` model that maps clinical entities and concepts (like drugs/ingredients) to RxNorm codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. This resolver model maps and extracts pharmaceutical actions (e.g analgesic, hypoglycemic) as well as treatments (e.g backache, diabetes) along with the RxNorm code resolved. Actions and treatments of the drugs are returned in `all_k_aux_labels` column.

 See [Model Card](https://nlp.johnsnowlabs.com/2022/04/25/sbiobertresolve_rxnorm_action_treatment_en_2_4.html) for details.

*Example* :

```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
      .setInputCols(["ner_chunk"])\
      .setOutputCol("sentence_embeddings")

rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_action_treatment", "en", "clinical/models") \
      .setInputCols(["ner_chunk", "sentence_embeddings"]) \
      .setOutputCol("rxnorm_code")\
      .setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        rxnorm_resolver])

lp_model = LightPipeline(pipelineModel)

text = ["Zita 200 mg", "coumadin 5 mg", 'avandia 4 mg']

result= lp_model.annotate(text)

```
Results* :
```
|    | ner_chunk     |   rxnorm_code | action                                  | treatment                          |
|---:|:--------------|--------------:|:----------------------------------------|------------------------------------|
|  0 | Zita 200 mg   |        104080 | ['Analgesic', 'Antacid', 'Antipyretic'] | ['Backache', 'Pain', 'Sore Throat']|
|  1 | coumadin 5 mg |        855333 | ['Anticoagulant']                       | ['Cerebrovascular Accident']       |
|  2 | avandia 4 mg  |        261242 | ['Drugs Used In Diabets','Hypoglycemic']| ['Diabetes Mellitus', ...]         |                                                                                              |
```

#### New RCT Classification Models and Pretrained Pipelines

We are releasing new **Randomized Clinical Trial (RCT)** classification models and pretrained pipelines that can classify the sections within the abstracts of scientific articles regarding randomized clinical trials (RCT).

+ Classification Models:
	+ `rct_binary_classifier_use` ([Models Hub page](https://nlp.johnsnowlabs.com/2022/04/24/rct_binary_classifier_use_en_3_0.html))
	+ `rct_binary_classifier_biobert` ([Models Hub page](https://nlp.johnsnowlabs.com/2022/04/25/rct_binary_classifier_biobert_en_3_0.html))
	+ `bert_sequence_classifier_binary_rct_biobert` ([Models Hub page](https://nlp.johnsnowlabs.com/2022/04/25/bert_sequence_classifier_binary_rct_biobert_en_3_0.html))

+ Pretrained Pipelines:
	+ `rct_binary_classifier_use_pipeline` ([Models Hub page](https://nlp.johnsnowlabs.com/2022/04/25/rct_binary_classifier_use_pipeline_en_3_0.html))
	+ `rct_binary_classifier_biobert_pipeline` ([Models Hub page](https://nlp.johnsnowlabs.com/2022/04/25/rct_binary_classifier_biobert_pipeline_en_3_0.html))
	+ `bert_sequence_classifier_binary_rct_biobert_pipeline` ([Models Hub page](https://nlp.johnsnowlabs.com/2022/04/25/bert_sequence_classifier_binary_rct_biobert_pipeline_en_3_0.html))

 *Classification Model Example* :

```python
...
use = UniversalSentenceEncoder.pretrained()\
        .setInputCols("document")\
        .setOutputCol("sentence_embeddings")

classifier_dl = ClassifierDLModel.pretrained('rct_binary_classifier_use', 'en', 'clinical/models')\
        .setInputCols(["sentence_embeddings"])\
        .setOutputCol("class")

use_clf_pipeline = Pipeline(
    stages = [
        document_assembler,
        use,
        classifier_dl
    ])

sample_text = """Abstract:Based on the American Society of Anesthesiologists' Practice Guidelines for Sedation and Analgesia by Non-Anesthesiologists (ASA-SED), a sedation training course aimed at improving medical safety was developed by the Japanese Association for Medical Simulation in 2011. This study evaluated the effect of debriefing on participants' perceptions of the essential points of the ASA-SED. A total of 38 novice doctors participated in the sedation training course during the research period. Of these doctors, 18 participated in the debriefing group, and 20 participated in non-debriefing group. Scoring of participants' guideline perceptions was conducted using an evaluation sheet (nine items, 16 points) created based on the ASA-SED. The debriefing group showed a greater perception of the ASA-SED, as reflected in the significantly higher scores on the evaluation sheet (median, 16 points) than the control group (median, 13 points; p < 0.05). No significant differences were identified before or during sedation, but the difference after sedation was significant (p < 0.05). Debriefing after sedation training courses may contribute to better perception of the ASA-SED, and may lead to enhanced attitudes toward medical safety during sedation and analgesia. """

result = use_clf_pipeline.transform(spark.createDataFrame([[sample_text]]).toDF("text"))

```

*Results* :
```
>> class: True
```

*Pretrained Pipeline Example* :

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("rct_binary_classifier_use_pipeline", "en", "clinical/models")
```

```
text = """Abstract:Based on the American Society of Anesthesiologists' Practice Guidelines for Sedation and Analgesia by Non-Anesthesiologists (ASA-SED), a sedation training course aimed at improving medical safety was developed by the Japanese Association for Medical Simulation in 2011. This study evaluated the effect of debriefing on participants' perceptions of the essential points of the ASA-SED. A total of 38 novice doctors participated in the sedation training course during the research period. Of these doctors, 18 participated in the debriefing group, and 20 participated in non-debriefing group. Scoring of participants' guideline perceptions was conducted using an evaluation sheet (nine items, 16 points) created based on the ASA-SED. The debriefing group showed a greater perception of the ASA-SED, as reflected in the significantly higher scores on the evaluation sheet (median, 16 points) than the control group (median, 13 points; p < 0.05). No significant differences were identified before or during sedation, but the difference after sedation was significant (p < 0.05). Debriefing after sedation training courses may contribute to better perception of the ASA-SED, and may lead to enhanced attitudes toward medical safety during sedation and analgesia. """

result = pipeline.annotate(text)
```

*Results* :

```
>> class: True
```

#### New Features
##### Add `getClasses()` attribute to `MedicalBertForTokenClassifier` and `MedicalBertForSequenceClassification`
Now you can use `getClasses()` method for checking the entity labels of  `MedicalBertForTokenClassifier` and `MedicalBertForSequenceClassification` like `MedicalNerModel`.

  ```python
  tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_ade", "en", "clinical/models")\
  	.setInputCols("token", "document")\
  	.setOutputCol("ner")\
  	.setCaseSensitive(True)\
  	.setMaxSentenceLength(512)

  tokenClassifier.getClasses()
  ```

  ```bash
  ['B-DRUG', 'I-ADE', 'I-DRUG', 'O', 'B-ADE']
  ```

##### Download the AnnotatorModels from the healthcare library using the Healthcare version instead of the open source version

Now we download the private models using the Healthcare version instead of the open source version (the pretrained models were used to be dependent on open source Spark NLP version before).

##### New functionality to download and extract clinical models from S3 via direct link.
Now, you can download clinical models from S3 via direct link directly by `downloadModelDirectly` method. See the [Models Hub Page](https://nlp.johnsnowlabs.com/models) to find out the download url of each model.

  ```python
  from sparknlp.pretrained import ResourceDownloader

  #The first argument is the path to the zip file and the second one is the folder.
  ResourceDownloader.downloadModelDirectly("clinical/models/assertion_dl_en_2.0.2_2.4_1556655581078.zip", "clinical/models")  
  ```

#### Core improvements:

##### Fix `MedicalNerModel` confidence scores when `setIncludeAllConfidenceScores` is `True`

A mismatch problem between the tag with the highest confidence score and the predicted tag in `MedicalNerModel` is resolved.

##### Graph_builder `relation_extraction` model file name extension problem with `auto` param

A naming problem which occurs while generating a graph for Relation Extraction via graph builder was resolved. Now, the TF graph is generated with the correct extension (`.pb`).

#### List of Recently Updated or Added Models

- ner_deid_generic_pt
- ner_deid_subentity_pt
- clinical_deidentification_pt
- sbiobertresolve_rxnorm_action_treatment
- rct_binary_classifier_use 
- rct_binary_classifier_biobert 
- bert_sequence_classifier_binary_rct_biobert 
- rct_binary_classifier_use_pipeline 
- rct_binary_classifier_biobert_pipeline 
- bert_sequence_classifier_binary_rct_biobert_pipeline 
- sbiobertresolve_ndc


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_5_0">Version 3.5.0</a>
    </li>
    <li>
        <strong>Version 3.5.1</strong>
    </li>
    <li>
        <a href="release_notes_3_5_2">Version 3.5.2</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_1">4.2.1</a></li>
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li class="active"><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>