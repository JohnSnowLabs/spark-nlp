---
layout: model
title: Detect Oncology-Specific Entities
author: John Snow Labs
name: ner_oncology
date: 2022-10-25
tags: [licensed, clinical, oncology, en, ner, biomarker, treatment]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts more than 40 oncology-related entities, including therapies, tests and staging.

Definitions of Predicted Entities:

- `Adenopathy`: Mentions of pathological findings of the lymph nodes.
- `Age`: All mention of ages, past or present, related to the patient or with anybody else.
- `Biomarker`: Biological molecules that indicate the presence or absence of cancer, or the type of cancer. Oncogenes are excluded from this category.
- `Biomarker_Result`: Terms or values that are identified as the result of a biomarkers.
- `Cancer_Dx`: Mentions of cancer diagnoses (such as "breast cancer") or pathological types that are usually used as synonyms for "cancer" (e.g. "carcinoma"). When anatomical references are present, they are included in the Cancer_Dx extraction.
- `Cancer_Score`: Clinical or imaging scores that are specific for cancer settings (e.g. "BI-RADS" or "Allred score").
- `Cancer_Surgery`: Terms that indicate surgery as a form of cancer treatment.
- `Chemotherapy`: Mentions of chemotherapy drugs, or unspecific words such as "chemotherapy".
- `Cycle_Coun`: The total number of cycles being administered of an oncological therapy (e.g. "5 cycles"). 
- `Cycle_Day`: References to the day of the cycle of oncological therapy (e.g. "day 5").
- `Cycle_Number`: The number of the cycle of an oncological therapy that is being applied (e.g. "third cycle").
- `Date`: Mentions of exact dates, in any format, including day number, month and/or year.
- `Death_Entity`: Words that indicate the death of the patient or someone else (including family members), such as "died" or "passed away".
- `Direction`: Directional and laterality terms, such as "left", "right", "bilateral", "upper" and "lower".
- `Dosage`: The quantity prescribed by the physician for an active ingredient.
- `Duration`: Words indicating the duration of a treatment (e.g. "for 2 weeks").
- `Frequency`: Words indicating the frequency of treatment administration (e.g. "daily" or "bid").
- `Gender`: Gender-specific nouns and pronouns (including words such as "him" or "she", and family members such as "father").
- `Grade`: All pathological grading of tumors (e.g. "grade 1") or degrees of cellular differentiation (e.g. "well-differentiated")
- `Histological_Type`: Histological variants or cancer subtypes, such as "papillary", "clear cell" or "medullary". 
- `Hormonal_Therapy`: Mentions of hormonal drugs used to treat cancer, or unspecific words such as "hormonal therapy".
- `Imaging_Test`: Imaging tests mentioned in texts, such as "chest CT scan".
- `Immunotherapy`: Mentions of immunotherapy drugs, or unspecific words such as "immunotherapy".
- `Invasion`: Mentions that refer to tumor invasion, such as "invasion" or "involvement". Metastases or lymph node involvement are excluded from this category.
- `Line_Of_Therapy`: Explicit references to the line of therapy of an oncological therapy (e.g. "first-line treatment").
- `Metastasis`: Terms that indicate a metastatic disease. Anatomical references are not included in these extractions.
- `Oncogene`: Mentions of genes that are implicated in the etiology of cancer.
- `Pathology_Result`: The findings of a biopsy from the pathology report that is not covered by another entity (e.g. "malignant ductal cells").
- `Pathology_Test`: Mentions of biopsies or tests that use tissue samples.
- `Performance_Status`: Mentions of performance status scores, such as ECOG and Karnofsky. The name of the score is extracted together with the result (e.g. "ECOG performance status of 4").
- `Race_Ethnicity`: The race and ethnicity categories include racial and national origin or sociocultural groups.
- `Radiotherapy`: Terms that indicate the use of Radiotherapy.
- `Response_To_Treatment`: Terms related to clinical progress of the patient related to cancer treatment, including "recurrence", "bad response" or "improvement".
- `Relative_Date`: Temporal references that are relative to the date of the text or to any other specific date (e.g. "yesterday" or "three years later").
- `Route`: Words indicating the type of administration route (such as "PO" or "transdermal").
- `Site_Bone`: Anatomical terms that refer to the human skeleton.
- `Site_Brain`: Anatomical terms that refer to the central nervous system (including the brain stem and the cerebellum).
- `Site_Breast`: Anatomical terms that refer to the breasts.
- `Site_Liver`: Anatomical terms that refer to the liver.
- `Site_Lung`: Anatomical terms that refer to the lungs.
- `Site_Lymph_Node`: Anatomical terms that refer to lymph nodes, excluding adenopathies.
- `Site_Other_Body_Part`: Relevant anatomical terms that are not included in the rest of the anatomical entities.
- `Smoking_Status`: All mentions of smoking related to the patient or to someone else.
- `Staging`: Mentions of cancer stage such as "stage 2b" or "T2N1M0". It also includes words such as "in situ", "early-stage" or "advanced".
- `Targeted_Therapy`: Mentions of targeted therapy drugs, or unspecific words such as "targeted therapy".
- `Tumor_Finding`: All nonspecific terms that may be related to tumors, either malignant or benign (for example: "mass", "tumor", "lesion", or "neoplasm").
- `Tumor_Size`: Size of the tumor, including numerical value and unit of measurement (e.g. "3 cm").
- `Unspecific_Therapy`: Terms that indicate a known cancer therapy but that is not specific to any other therapy entity (e.g. "chemoradiotherapy" or "adjuvant therapy").



## Predicted Entities

`Histological_Type`, `Direction`, `Staging`, `Cancer_Score`, `Imaging_Test`, `Cycle_Number`, `Tumor_Finding`, `Site_Lymph_Node`, `Invasion`, `Response_To_Treatment`, `Smoking_Status`, `Tumor_Size`, `Cycle_Count`, `Adenopathy`, `Age`, `Biomarker_Result`, `Unspecific_Therapy`, `Site_Breast`, `Chemotherapy`, `Targeted_Therapy`, `Radiotherapy`, `Performance_Status`, `Pathology_Test`, `Site_Other_Body_Part`, `Cancer_Surgery`, `Line_Of_Therapy`, `Pathology_Result`, `Hormonal_Therapy`, `Site_Bone`, `Biomarker`, `Immunotherapy`, `Cycle_Day`, `Frequency`, `Route`, `Duration`, `Death_Entity`, `Metastasis`, `Site_Liver`, `Cancer_Dx`, `Grade`, `Date`, `Site_Lung`, `Site_Brain`, `Relative_Date`, `Race_Ethnicity`, `Gender`, `Oncogene`, `Dosage`, `Radiation_Dose`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_en_4.0.0_3.0_1666718178718.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_en_4.0.0_3.0_1666718178718.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")                

ner = MedicalNerModel.pretrained("ner_oncology", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter])

data = spark.createDataFrame([["The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to the residual breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter))    

val data = Seq("The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to the residual breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
| chunk                          | ner_label             |
|:-------------------------------|:----------------------|
| left                           | Direction             |
| mastectomy                     | Cancer_Surgery        |
| axillary lymph node dissection | Cancer_Surgery        |
| left                           | Direction             |
| breast cancer                  | Cancer_Dx             |
| twenty years ago               | Relative_Date         |
| tumor                          | Tumor_Finding         |
| positive                       | Biomarker_Result      |
| ER                             | Biomarker             |
| PR                             | Biomarker             |
| radiotherapy                   | Radiotherapy          |
| breast                         | Site_Breast           |
| cancer                         | Cancer_Dx             |
| recurred                       | Response_To_Treatment |
| right                          | Direction             |
| lung                           | Site_Lung             |
| metastasis                     | Metastasis            |
| 13 years later                 | Relative_Date         |
| adriamycin                     | Chemotherapy          |
| 60 mg/m2                       | Dosage                |
| cyclophosphamide               | Chemotherapy          |
| 600 mg/m2                      | Dosage                |
| six courses                    | Cycle_Count           |
| first line                     | Line_Of_Therapy       |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|34.6 MB|
|Dependencies:|embeddings_clinical|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
                label    tp   fp   fn  total  precision  recall   f1
    Histological_Type   339   75  114    453       0.82    0.75 0.78
            Direction   832  163  152    984       0.84    0.85 0.84
              Staging   229   31   29    258       0.88    0.89 0.88
         Cancer_Score    37    8   25     62       0.82    0.60 0.69
         Imaging_Test  2027  214  177   2204       0.90    0.92 0.91
         Cycle_Number    73   29   24     97       0.72    0.75 0.73
        Tumor_Finding  1114   64  143   1257       0.95    0.89 0.91
      Site_Lymph_Node   491   53   60    551       0.90    0.89 0.90
             Invasion   158   36   23    181       0.81    0.87 0.84
Response_To_Treatment   431  149  165    596       0.74    0.72 0.73
       Smoking_Status    66   18    2     68       0.79    0.97 0.87
           Tumor_Size  1050  112   79   1129       0.90    0.93 0.92
          Cycle_Count   177   62   53    230       0.74    0.77 0.75
           Adenopathy    67   12   29     96       0.85    0.70 0.77
                  Age   930   33   19    949       0.97    0.98 0.97
     Biomarker_Result  1160  169  285   1445       0.87    0.80 0.84
   Unspecific_Therapy   198   86   80    278       0.70    0.71 0.70
          Site_Breast   125   15   22    147       0.89    0.85 0.87
         Chemotherapy   814   55   65    879       0.94    0.93 0.93
     Targeted_Therapy   195   27   33    228       0.88    0.86 0.87
         Radiotherapy   276   29   34    310       0.90    0.89 0.90
   Performance_Status   121   17   14    135       0.88    0.90 0.89
       Pathology_Test   888  296  162   1050       0.75    0.85 0.79
 Site_Other_Body_Part   909  275  592   1501       0.77    0.61 0.68
       Cancer_Surgery   693  119  126    819       0.85    0.85 0.85
      Line_Of_Therapy   101   11    5    106       0.90    0.95 0.93
     Pathology_Result   655  279  487   1142       0.70    0.57 0.63
     Hormonal_Therapy   169    4   16    185       0.98    0.91 0.94
            Site_Bone   264   81   49    313       0.77    0.84 0.80
            Biomarker  1259  238  256   1515       0.84    0.83 0.84
        Immunotherapy   103   47   25    128       0.69    0.80 0.74
            Cycle_Day   200   36   48    248       0.85    0.81 0.83
            Frequency   354   27   73    427       0.93    0.83 0.88
                Route    91   15   22    113       0.86    0.81 0.83
             Duration   625  161  136    761       0.80    0.82 0.81
         Death_Entity    34    2    4     38       0.94    0.89 0.92
           Metastasis   353   18   17    370       0.95    0.95 0.95
           Site_Liver   189   64   45    234       0.75    0.81 0.78
            Cancer_Dx  1301  103   93   1394       0.93    0.93 0.93
                Grade   190   27   46    236       0.88    0.81 0.84
                 Date   807   21   24    831       0.97    0.97 0.97
            Site_Lung   469  110   90    559       0.81    0.84 0.82
           Site_Brain   221   64   58    279       0.78    0.79 0.78
        Relative_Date  1211  401  111   1322       0.75    0.92 0.83
       Race_Ethnicity    57    8    5     62       0.88    0.92 0.90
               Gender  1247   17    7   1254       0.99    0.99 0.99
             Oncogene   345   83  104    449       0.81    0.77 0.79
               Dosage   900   30  160   1060       0.97    0.85 0.90
       Radiation_Dose   108    5   18    126       0.96    0.86 0.90
            macro_avg 24653 3999 4406  29059       0.85    0.84 0.84
            micro_avg 24653 3999 4406  29059       0.86    0.85 0.85
```
