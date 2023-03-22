---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.3.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_3_2
key: docs-licensed-release-notes
modify_date: 2023-03-18
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">


## 4.3.2

#### Highlights

+ Welcoming BioGPT (Generative pre-trained transformer for biomedical text generation and mining) to Spark NLP, with a faster inference and better memory management.
+ New `MedicalQuestionAnswering` annotator based on BioGPT to answer questions from PubMed abstracts
+ Crossing 1000+ healthcare specific pretrained models & pipelines in the Model hub
+ Running obfuscation and deidentification at the same time, based on selected entities in one pass
+ Core improvements and bug fixes
    - New features added to  `NameChunkObfuscation` module
    - More flexibility for `setAgeRanges` in `DeIdentification`
    - Added new sub-module to the ALAB module for reviewing annotations and spotting label errors easily
    - Added `ner_jsl` model label definitions to the model cards
    - More flexibility in `ocr_nlp_processor` with new parameters for the OCR pipeline
    - Updated 120+ clinical pipelines to make them compatible with all PySpark versions
+ New and updated notebooks
+ New and updated demos
    - [Medical Question Answering](https://demo.johnsnowlabs.com/healthcare/BIOGPT_MEDICAL_QUESTION_ANSWERING/) demo
    - [Social Determinants of Health Behaviour Problems](https://demo.johnsnowlabs.com/healthcare/NER_SDOH_BEHAVIOURS_PROBLEMS/) demo
    - [Social Determinants of Health Access Status](https://demo.johnsnowlabs.com/healthcare/NER_SDOH_ACCESS/) demo
    - [Voice of The Patients](https://demo.johnsnowlabs.com/healthcare/VOICE_OF_THE_PATIENTS/) demo
+ New blogposts
+ 30+ new clinical models and pipelines added & updated in total

</div><div class="h3-box" markdown="1">

#### Welcoming BioGPT (Generative Pre-Trained Transformer For Biomedical Text Generation and Mining) to Spark NLP

`BioGPT` is a domain-specific generative pre-trained Transformer language model for biomedical text generation and mining. `BioGPT` follows the Transformer language model backbone, and is pre-trained on 15M PubMed abstracts from scratch. Experiments demonstrate that `BioGPT` achieves better performance compared with baseline methods and other well-performing methods across all the tasks. Read more at [the official paper](https://arxiv.org/abs/2210.10341).

We ported `BioGPT` (`BioGPT-QA-PubMedQA-BioGPT`) into Spark NLP for Healthcare with better inference speed and memory optimization.



</div><div class="h3-box" markdown="1">

#### New `MedicalQuestionAnswering` Annotator Based On BioGPT To Answer Questions From PubMed Abstracts

New [medical_qa_biogpt](https://nlp.johnsnowlabs.com/2023/03/09/medical_qa_biogpt_en.html) model is based on the original `BioGPT-QA-PubMedQA-BioGPT` model (trained with Pubmed abstracts) can generate two types of answers, *short* and *long*.

- The first type of question is `"short"` and is designed to elicit a simple, concise answer that is typically one of three options: `yes`, `no`, or `maybe`.

- The second type of question is `"long"` and intended to prompt a more detailed response. Unlike the `short` questions, which are generally answerable with a single word, `long` questions require a more thoughtful and comprehensive response.

Overall, the distinction between *short* and *long* questions is based on the complexity of the answers they are meant to elicit. *Short* questions are used when a quick and simple answer is sufficient, while *long* questions are used when a more detailed and nuanced response is required.

```python
med_qa = MedicalQuestionAnswering.pretrained("medical_qa_biogpt","en","clinical/models")\
    .setInputCols(["document_question", "document_context"])\
    .setOutputCol("answer")\
    .setMaxNewTokens(30)\
    .setTopK(1)\
    .setQuestionType("long") # "short"

pipeline = Pipeline(stages=[document_assembler, med_qa])

paper_abstract = "The visual indexing theory proposed by Zenon Pylyshyn (Cognition, 32, 65–97, 1989) predicts that visual attention mechanisms are employed when mental images are projected onto a visual scene. Recent eye-tracking studies have supported this hypothesis by showing that people tend to look at empty places where requested information has been previously presented. However, it has remained unclear to what extent this behavior is related to memory performance. The aim of the present study was to explore whether the manipulation of spatial attention can facilitate memory retrieval. In two experiments, participants were asked first to memorize a set of four objects and then to determine whether a probe word referred to any of the objects. The results of both experiments indicate that memory accuracy is not affected by the current focus of attention and that all the effects of directing attention to specific locations on response times can be explained in terms of stimulus–stimulus and stimulus–response spatial compatibility."
```

*Result for `long` answer*:

```bash
Question ["What is the effect of directing attention on memory?"]
Answer ["the results of the present study suggest that the visual indexing theory does not fully explain the effects of spatial attention on memory performance."]
```

*Result for `short` answer*:

```bash
Question ["Does directing attention improve memory for items?"]
Answer ["no"]
```

You can check the [Medical Question Answering Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/healthcare-nlp/31.Medical_Question_Answering.ipynb) for more examples and see the [Medical Question Answering](https://demo.johnsnowlabs.com/healthcare/BIOGPT_MEDICAL_QUESTION_ANSWERING/) demo.



</div><div class="h3-box" markdown="1">

#### Crossing 1000+ Healthcare Specific Pretrained Models & Pipelines In Models Hub

We just crossed **1000+ healthcare specific pretrained models & pipelines** in the [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)!




</div><div class="h3-box" markdown="1">

#### Running Obfuscation and Deidentification At The Same Time, Based On Selected Entities In One Pass

The `DeIdentification()` annotator has been enhanced with the inclusion of **multi-mode functionality**. Users are required to define a dictionary that contains the policies which will be applied to the labels and save it as a JSON file. Then multi-mode functionality can be utilized in the de-identification process by providing the path of the JSON file to the `setSelectiveObfuscationModes()` parameter. If the entities are not provided in the JSON file, they will be deidentified according to the `setMode()` as default.


Example JSON file :

```bash
sample_deid = {
  	"obfuscate": ["PHONE"],
  	"mask_entity_labels": ["ID"],
  	"skip": ["DATE"],
  	"mask_same_length_chars": ["NAME"],
  	"mask_fixed_length_chars": ["ZIP", "LOCATION"]
    }
```

Description of possible modes to enable multi-mode deidentification:


 * `obfuscate`: Replace the values with random values.
 * `mask_same_length_chars`: Replace the name with the minus two same lengths asterisk, plus one bracket on both ends.
 * `mask_entity_labels`: Replace the values with the entity labels.
 * `mask_fixed_length_chars`: Replace the name with the asterisk with fixed length. You can also invoke `setFixedMaskLength()`.
 * `skip`: Skip the entities (intact).


*Example:*

```python
...
deid = DeIdentification() \
      .setInputCols(["sentence", "token", "ner_chunk"]) \
      .setOutputCol("deidentified") \
      .setMode("obfuscate")\
      .setSelectiveObfuscationModesPath("sample_deid.json")\
      .setSameLengthFormattedEntities(["PHONE"])
      
text = "Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson Ora , M.R # 7194334 Date : 01/13/93 . PCP : Oliveira , 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital , 0295 Keats Street , Phone 55-555-5555 ."
```



*Result:*

```bash
[Record date : 2093-01-13 , [********] , M.D . , Name : [*************] , M.R \# <ID>, Date : 01/13/93 . PCP : [******] , <AGE> years-old , Record date : 2079-11-09 . ******* , ******* , Phone 98-496-9970 ]
```

- `DATE` entities were skipped: `2093-01-13` => `2093-01-13`, `01/13/93`=> `01/13/93`
- `PHONE` entity was obfuscated with fake phone number: `55-555-5555` => `98-496-9970`
- `ID` entity was masked with ID tag: `7194334` => `<ID>`
- `NAME` entities were masked with same original lenght: `David Hale` = > `[********]`, `Hendrickson Ora` => `[*************]`
- `LOCATION` entities were masked with fixed lenght: `Cocke County Baptist Hospital` => `*******` , `0295 Keats Street` => `*******`


</div><div class="h3-box" markdown="1">

#### Core Improvements and Bug Fixes


- New features added to  `NameChunkObfuscation` module
- More flexibility for `setAgeRanges` in `DeIdentification`
- Adding new sub-module to the ALAB module to review annotation and spot label errors easily
- Added `ner_jsl` model label definitions to the [model card](https://nlp.johnsnowlabs.com/2022/10/19/ner_jsl_en.html)
- More flexibility in `ocr_nlp_processor` with new parameters for the OCR pipeline, please see [Spark OCR Utility Module](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/5.3.Spark_OCR_Utility_Module.ipynb)
- Updated 120+ clinical pipelines to make them compatible with all PySpark versions


</div><div class="h3-box" markdown="1">

#### New and Updated Notebooks

- New [Medical Question Answering Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/healthcare-nlp/31.Medical_Question_Answering.ipynb) for showing how medical question answering can be used with new `MedicalQuestionAnswering` annotator.
- Updated [Clinical DeIdentification Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb) with latest updates.

</div><div class="h3-box" markdown="1">

#### New and Updated Demos

+ [Medical Question Answering](https://demo.johnsnowlabs.com/healthcare/BIOGPT_MEDICAL_QUESTION_ANSWERING/) demo
+ [Social Determinants of Health Behaviour Problems](https://demo.johnsnowlabs.com/healthcare/NER_SDOH_BEHAVIOURS_PROBLEMS/) demo
+ [Social Determinants of Health Access Status](https://demo.johnsnowlabs.com/healthcare/NER_SDOH_ACCESS/) demo
+ [Voice of The Patients](https://demo.johnsnowlabs.com/healthcare/VOICE_OF_THE_PATIENTS/) demo


</div><div class="h3-box" markdown="1">

#### New Blogposts

- [Extract Social Determinants of Health Entities From Clinical Text with Spark NLP](https://medium.com/john-snow-labs/extract-social-determinants-of-health-entities-from-clinical-text-with-spark-nlp-542a9a4e0ffc)
- [Extract Clinical Entities From Patient Forums with Healthcare NLP](https://www.johnsnowlabs.com/extract-clinical-entities-from-patient-forums-with-healthcare-nlp/)
- [Mapping Rxnorm and NDC Codes to the National Institute of Health (NIH) Drug Brand Names with Spark NLP](https://medium.com/john-snow-labs/mapping-rxnorm-and-ndc-codes-to-the-nih-drug-brand-names-with-spark-nlp-e10eeb7e122c)
- [Format Consistency For Entity Obfuscation In De-Identification with Spark NLP](https://medium.com/john-snow-labs/format-consistency-for-entity-obfuscation-in-de-identification-with-spark-nlp-9d850a25e455)

</div><div class="h3-box" markdown="1">

#### 30+ New Clinical Models and Pipelines Added & Updated in Total

+ `biogpt_pubmed_qa`
+ 30+ new clinical ner pipelines


</div><div class="h3-box" markdown="1">

For all Spark NLP for Healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}
