---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Export Data
permalink: /docs/en/alab/export
key: docs-training
modify_date: "2020-11-18"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

The completions and predictions are stored in a database for fast search and access. Completions and predictions can be exported into the formats described below.

## JSON
You can convert and export the completions and predictions to json format by using the JSON option on the **Export** page. 
The obtained format is the following:

```bash
[
  {
    "completions": [],
    "predictions": [
      {
        "created_username": "SparkNLP Pre-annotation",
        "result": [
          {
            "from_name": "label",
            "id": "7HGzTLkNUA",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 3554,
              "labels": [
                "Symptom_Name"
              ],
              "start": 3548,
              "text": "snores"
            }
          }
        ],
        "created_ago": "2020-11-09T14:44:57.713743Z",
        "id": 2001
      }
    ],
    "created_at": "2020-11-09 14:41:39",
    "created_by": "admin",
    "data": {
      "text": "Cardiovascular / Pulmonary\nSample Name: Angina - Consult\nDescription: Patient had a recurrent left arm pain after her stent, three days ago, and this persisted after two sublingual nitroglycerin.\n(Medical Transcription Sample Report)\nHISTORY OF PRESENT ILLNESS: The patient is a 68-year-old woman whom I have been following, who has had angina. In any case today, she called me because she had a recurrent left arm pain after her stent, three days ago, and this persisted after two sublingual nitroglycerin when I spoke to her.",
      "title": "sample_document3.txt",
      "pre_annotation": true
    },
    "id": 2
  }]
```
## CSV
Results are stored in comma-separated tabular file with column names specified by "from_name" "to_name" values

## TSV
Results are stored in tab-separated tabular file with column names specified by "from_name" "to_name" values

## CONLL2003

The CONLL export feature generates a single output file, containing all available completios for all the tasks in the project. The resulting file has the following format: 
```bash
-DOCSTART- -X- O
Sample -X- _ O
Type -X- _ O
Medical -X- _ O
Specialty: -X- _ O
Endocrinology -X- _ O

Sample -X- _ O
Name: -X- _ O
Diabetes -X- _ B-Diagnosis
Mellitus -X- _ I-Diagnosis
Followup -X- _ O

Description: -X- _ O
Return -X- _ O
visit -X- _ O
to -X- _ O
the -X- _ O
endocrine -X- _ O
clinic -X- _ O
for -X- _ O
followup -X- _ O
management -X- _ O
of -X- _ O
type -X- _ O
1 -X- _ O
diabetes -X- _ O
mellitus -X- _ O
Plan -X- _ O
today -X- _ O
is -X- _ O
to -X- _ O
make -X- _ O
adjustments -X- _ O
to -X- _ O
her -X- _ O
pump -X- _ O
based -X- _ O
on -X- _ O
a -X- _ O
total -X- _ O
daily -X- _ B-FREQUENCY
dose -X- _ O
of -X- _ O
90 -X- _ O
units -X- _ O
of -X- _ O
insulin -X- _ O
…
```

User can specify if only starred completions should be included in the output file by checking "Only ground truth" option before generating the export.

## Allow the export of tasks without completions

Previous versions of the Annotation Lab only allowed the export of tasks that contained completions. From version 2.8.0 on, the tasks without any completions can be exported as this can be necessary for cloning projects. In the case where only tasks with completions are required in the export, users can enable the “Exclude tasks without Completions” option on the export page. 

 ![export-page](https://user-images.githubusercontent.com/26042994/154637982-55872de3-85e2-4aaf-be4d-7e8c1d59417d.png)