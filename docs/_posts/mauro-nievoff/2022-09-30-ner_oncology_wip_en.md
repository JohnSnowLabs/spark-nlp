---
layout: model
title: Detect Oncology-Specific Entities
author: John Snow Labs
name: ner_oncology_wip
date: 2022-09-30
tags: [licensed, clinical, oncology, en, ner, biomarker, treatment]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.0.0
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

`Histological_Type`, `Direction`, `Staging`, `Cancer_Score`, `Imaging_Test`, `Cycle_Number`, `Tumor_Finding`, `Site_Lymph_Node`, `Invasion`, `Response_To_Treatment`, `Smoking_Status`, `Tumor_Size`, `Cycle_Count`, `Adenopathy`, `Age`, `Biomarker_Result`, `Unspecific_Therapy`, `Site_Breast`, `Chemotherapy`, `Targeted_Therapy`, `Radiotherapy`, `Performance_Status`, `Pathology_Test`, `Site_Other_Body_Part`, `Cancer_Surgery`, `Line_Of_Therapy`, `Pathology_Result`, `Hormonal_Therapy`, `Site_Bone`, `Biomarker`, `Immunotherapy`, `Cycle_Day`, `Frequency`, `Route`, `Duration`, `Death_Entity`, `Metastasis`, `Site_Liver`, `Cancer_Dx`, `Grade`, `Date`, `Site_Lung`, `Site_Brain`, `Relative_Date`, `Race_Ethnicity`, `Gender`, `Oncogene`, `Dosage`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_wip_en_4.0.0_3.0_1664556885893.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_wip_en_4.0.0_3.0_1664556885893.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models") \
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

data = spark.createDataFrame([["The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to her breast.The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models")
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
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to her breast.
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
|Model Name:|ner_oncology_wip|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|992.6 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
                label      tp     fp     fn   total  precision  recall   f1
    Histological_Type   200.0   60.0  143.0   343.0       0.77    0.58 0.66
            Direction   602.0  126.0  132.0   734.0       0.83    0.82 0.82
              Staging   160.0   17.0   56.0   216.0       0.90    0.74 0.81
         Cancer_Score    17.0    0.0   42.0    59.0       1.00    0.29 0.45
         Imaging_Test  1534.0  192.0  175.0  1709.0       0.89    0.90 0.89
         Cycle_Number    46.0   14.0   27.0    73.0       0.77    0.63 0.69
        Tumor_Finding   834.0   52.0  108.0   942.0       0.94    0.89 0.91
      Site_Lymph_Node   414.0   50.0   52.0   466.0       0.89    0.89 0.89
             Invasion    89.0    7.0   44.0   133.0       0.93    0.67 0.78
Response_To_Treatment   225.0   70.0  204.0   429.0       0.76    0.52 0.62
       Smoking_Status    48.0   15.0    6.0    54.0       0.76    0.89 0.82
           Tumor_Size   771.0  133.0   81.0   852.0       0.85    0.90 0.88
          Cycle_Count   143.0   59.0   32.0   175.0       0.71    0.82 0.76
           Adenopathy    27.0    8.0   17.0    44.0       0.77    0.61 0.68
                  Age   655.0   18.0   41.0   696.0       0.97    0.94 0.96
     Biomarker_Result   845.0  281.0  261.0  1106.0       0.75    0.76 0.76
   Unspecific_Therapy   131.0  168.0  103.0   234.0       0.44    0.56 0.49
          Site_Breast    69.0    6.0   49.0   118.0       0.92    0.58 0.72
         Chemotherapy   475.0   36.0  143.0   618.0       0.93    0.77 0.84
     Targeted_Therapy   139.0    8.0   40.0   179.0       0.95    0.78 0.85
         Radiotherapy   192.0   12.0   30.0   222.0       0.94    0.86 0.90
   Performance_Status    60.0   13.0   40.0   100.0       0.82    0.60 0.69
       Pathology_Test   631.0  154.0  178.0   809.0       0.80    0.78 0.79
 Site_Other_Body_Part   663.0  297.0  438.0  1101.0       0.69    0.60 0.64
       Cancer_Surgery   542.0  139.0  103.0   645.0       0.80    0.84 0.82
      Line_Of_Therapy    79.0   11.0    8.0    87.0       0.88    0.91 0.89
     Pathology_Result   546.0  369.0  309.0   855.0       0.60    0.64 0.62
     Hormonal_Therapy    82.0    1.0   34.0   116.0       0.99    0.71 0.82
            Site_Bone   166.0   52.0   66.0   232.0       0.76    0.72 0.74
            Biomarker   899.0  342.0  212.0  1111.0       0.72    0.81 0.76
        Immunotherapy    87.0   52.0   24.0   111.0       0.63    0.78 0.70
            Cycle_Day   142.0   28.0   41.0   183.0       0.84    0.78 0.80
            Frequency   295.0   27.0   54.0   349.0       0.92    0.85 0.88
                Route    50.0    4.0   47.0    97.0       0.93    0.52 0.66
             Duration   384.0   81.0  163.0   547.0       0.83    0.70 0.76
         Death_Entity    21.0    1.0    8.0    29.0       0.95    0.72 0.82
           Metastasis   274.0   13.0   15.0   289.0       0.95    0.95 0.95
           Site_Liver   125.0  142.0   45.0   170.0       0.47    0.74 0.57
            Cancer_Dx   938.0  120.0  128.0  1066.0       0.89    0.88 0.88
                Grade   119.0   19.0   79.0   198.0       0.86    0.60 0.71
                 Date   614.0   28.0   15.0   629.0       0.96    0.98 0.97
            Site_Lung   285.0   66.0  158.0   443.0       0.81    0.64 0.72
           Site_Brain   149.0   40.0   51.0   200.0       0.79    0.74 0.77
        Relative_Date   853.0  351.0  111.0   964.0       0.71    0.88 0.79
       Race_Ethnicity    41.0    7.0   10.0    51.0       0.85    0.80 0.83
               Gender   933.0   14.0    8.0   941.0       0.99    0.99 0.99
             Oncogene   198.0   53.0  159.0   357.0       0.79    0.55 0.65
               Dosage   675.0   81.0  152.0   827.0       0.89    0.82 0.85
       Radiation_Dose    79.0   20.0   17.0    96.0       0.80    0.82 0.81
            macro_avg 17546.0 3857.0 4459.0 22005.0       0.83    0.75 0.78
            micro_avg     NaN    NaN    NaN     NaN       0.83    0.80 0.81
```