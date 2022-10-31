---
layout: docs
comment: no
header: true
seotitle: Annotation Lab | John Snow Labs
title: Export
permalink: /docs/en/alab/export
key: docs-training
modify_date: "2022-10-31"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: annotation-lab
---

<style>
bl {
  font-weight: 400;
}

pre {
  max-height: 500px;
}
</style>

## Supported Formats

The completions and predictions are stored in a database for fast search and access. Completions and predictions can be exported into the formats described below.

### JSON
You can convert and export the completions and predictions to JSON format using the JSON option on the Export page. 
The obtained format is the following:

```bash
[
  {
    "completions": [
      {
        "created_username": "eric",
        "created_ago": "2022-10-29T14:42:50.867Z",
        "lead_time": 82,
        "result": [
          {
            "value": {
              "start": 175,
              "end": 187,
              "text": "tuberculosis",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.9524
            },
            "id": "zgam2AbdmY",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 213,
              "end": 239,
              "text": "Mycobacterium tuberculosis",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.904775
            },
            "id": "1v76SqlWtj",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 385,
              "end": 394,
              "text": "pneumonia",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.91655
            },
            "id": "CURKae4Eca",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 436,
              "end": 449,
              "text": "Streptococcus",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.9157500000000001
            },
            "id": "cM5BvAsZL4",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 454,
              "end": 465,
              "text": "Pseudomonas",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.91495
            },
            "id": "KGOLhb8OPV",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 532,
              "end": 540,
              "text": "Shigella",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.91655
            },
            "id": "JCIhVQTDZl",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 542,
              "end": 555,
              "text": "Campylobacter",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.9163
            },
            "id": "CkxrbwvFzb",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 561,
              "end": 571,
              "text": "Salmonella",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.9164000000000001
            },
            "id": "c6ev6McH4Z",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 623,
              "end": 630,
              "text": "tetanus",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.97
            },
            "id": "9ZmEaJnqKG",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 632,
              "end": 645,
              "text": "typhoid fever",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.976675
            },
            "id": "Uo5CWzdd1S",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 647,
              "end": 657,
              "text": "diphtheria",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.9737
            },
            "id": "7nc71jXT3P",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 659,
              "end": 667,
              "text": "syphilis",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.97355
            },
            "id": "nIKfsOWNyE",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 673,
              "end": 689,
              "text": "Hansen's disease",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.899025
            },
            "id": "SyuVYMn7ax",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 30,
              "end": 38,
              "text": "bacteria",
              "labels": [
                "Pathogen"
              ],
              "confidence": 1
            },
            "id": "lq7qtJj1yX",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 98,
              "end": 106,
              "text": "bacteria",
              "labels": [
                "Pathogen"
              ],
              "confidence": 1
            },
            "id": "kxaB_gMstN",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          }
        ],
        "honeypot": true,
        "copied_from": "prediction: 11001",
        "id": 11001,
        "confidence_range": [
          0,
          1
        ],
        "copy": true,
        "cid": "11001",
        "data_type": "prediction",
        "updated_at": "2022-10-29T15:13:03.445569Z",
        "updated_by": "eric",
        "submitted_at": "2022-10-30T20:57:54.303"
      },
      {
        "created_username": "jenny",
        "created_ago": "2022-10-29T15:03:51.669Z",
        "lead_time": 0,
        "result": [
          {
            "value": {
              "start": 175,
              "end": 187,
              "text": "tuberculosis",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.9524
            },
            "id": "zgam2AbdmY",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 213,
              "end": 239,
              "text": "Mycobacterium tuberculosis",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.904775
            },
            "id": "1v76SqlWtj",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 385,
              "end": 394,
              "text": "pneumonia",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.91655
            },
            "id": "CURKae4Eca",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 436,
              "end": 449,
              "text": "Streptococcus",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.9157500000000001
            },
            "id": "cM5BvAsZL4",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 454,
              "end": 465,
              "text": "Pseudomonas",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.91495
            },
            "id": "KGOLhb8OPV",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 532,
              "end": 540,
              "text": "Shigella",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.91655
            },
            "id": "JCIhVQTDZl",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 542,
              "end": 555,
              "text": "Campylobacter",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.9163
            },
            "id": "CkxrbwvFzb",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 561,
              "end": 571,
              "text": "Salmonella",
              "labels": [
                "Pathogen"
              ],
              "confidence": 0.9164000000000001
            },
            "id": "c6ev6McH4Z",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 623,
              "end": 630,
              "text": "tetanus",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.97
            },
            "id": "9ZmEaJnqKG",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 632,
              "end": 645,
              "text": "typhoid fever",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.976675
            },
            "id": "Uo5CWzdd1S",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 647,
              "end": 657,
              "text": "diphtheria",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.9737
            },
            "id": "7nc71jXT3P",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 659,
              "end": 667,
              "text": "syphilis",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.97355
            },
            "id": "nIKfsOWNyE",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          },
          {
            "value": {
              "start": 673,
              "end": 689,
              "text": "Hansen's disease",
              "labels": [
                "MedicalCondition"
              ],
              "confidence": 0.899025
            },
            "id": "SyuVYMn7ax",
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
          }
        ],
        "honeypot": true,
        "confidence_range": [
          0,
          1
        ],
        "submitted_at": "2022-10-29T20:48:51.669",
        "id": 11002
      }
    ],
    "predictions": [
      {
        "created_username": "SparkNLP Pre-annotation",
        "result": [
          {
            "from_name": "label",
            "id": "zgam2AbdmY",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 187,
              "labels": [
                "MedicalCondition"
              ],
              "start": 175,
              "text": "tuberculosis",
              "confidence": "0.9524"
            }
          },
          {
            "from_name": "label",
            "id": "1v76SqlWtj",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 239,
              "labels": [
                "Pathogen"
              ],
              "start": 213,
              "text": "Mycobacterium tuberculosis",
              "confidence": "0.904775"
            }
          },
          {
            "from_name": "label",
            "id": "CURKae4Eca",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 394,
              "labels": [
                "MedicalCondition"
              ],
              "start": 385,
              "text": "pneumonia",
              "confidence": "0.91655"
            }
          },
          {
            "from_name": "label",
            "id": "cM5BvAsZL4",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 449,
              "labels": [
                "Pathogen"
              ],
              "start": 436,
              "text": "Streptococcus",
              "confidence": "0.9157500000000001"
            }
          },
          {
            "from_name": "label",
            "id": "KGOLhb8OPV",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 465,
              "labels": [
                "Pathogen"
              ],
              "start": 454,
              "text": "Pseudomonas",
              "confidence": "0.91495"
            }
          },
          {
            "from_name": "label",
            "id": "JCIhVQTDZl",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 540,
              "labels": [
                "Pathogen"
              ],
              "start": 532,
              "text": "Shigella",
              "confidence": "0.91655"
            }
          },
          {
            "from_name": "label",
            "id": "CkxrbwvFzb",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 555,
              "labels": [
                "Pathogen"
              ],
              "start": 542,
              "text": "Campylobacter",
              "confidence": "0.9163"
            }
          },
          {
            "from_name": "label",
            "id": "c6ev6McH4Z",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 571,
              "labels": [
                "Pathogen"
              ],
              "start": 561,
              "text": "Salmonella",
              "confidence": "0.9164000000000001"
            }
          },
          {
            "from_name": "label",
            "id": "9ZmEaJnqKG",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 630,
              "labels": [
                "MedicalCondition"
              ],
              "start": 623,
              "text": "tetanus",
              "confidence": "0.97"
            }
          },
          {
            "from_name": "label",
            "id": "Uo5CWzdd1S",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 645,
              "labels": [
                "MedicalCondition"
              ],
              "start": 632,
              "text": "typhoid fever",
              "confidence": "0.976675"
            }
          },
          {
            "from_name": "label",
            "id": "7nc71jXT3P",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 657,
              "labels": [
                "MedicalCondition"
              ],
              "start": 647,
              "text": "diphtheria",
              "confidence": "0.9737"
            }
          },
          {
            "from_name": "label",
            "id": "nIKfsOWNyE",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 667,
              "labels": [
                "MedicalCondition"
              ],
              "start": 659,
              "text": "syphilis",
              "confidence": "0.97355"
            }
          },
          {
            "from_name": "label",
            "id": "SyuVYMn7ax",
            "source": "$text",
            "to_name": "text",
            "type": "labels",
            "value": {
              "end": 689,
              "labels": [
                "MedicalCondition"
              ],
              "start": 673,
              "text": "Hansen's disease",
              "confidence": "0.899025"
            }
          }
        ],
        "created_ago": "2022-10-29T14:07:58.553246Z",
        "id": 11001
      }
    ],
    "created_at": "2022-10-29 14:07:12",
    "created_by": "admin",
    "data": {
      "text": "Although the vast majority of bacteria are harmless or beneficial to one's body, a few pathogenic bacteria can cause infectious diseases. The most common bacterial disease is tuberculosis, caused by the bacterium Mycobacterium tuberculosis, which affects about 2 million people mostly in sub-Saharan Africa. Pathogenic bacteria contribute to other globally important diseases, such as pneumonia, which can be caused by bacteria such as Streptococcus and Pseudomonas, and foodborne illnesses, which can be caused by bacteria such as Shigella, Campylobacter, and Salmonella. Pathogenic bacteria also cause infections such as tetanus, typhoid fever, diphtheria, syphilis, and Hansen's disease. They typically range between 1 and 5 micrometers in length.",
      "title": "cord19-11.txt"
    },
    "id": 11
  }
]
```

### CSV
Results are stored in a comma-separated tabular file with column names specified by "from_name" and "to_name" values.

### TSV
Results are stored in a tab-separated tabular file with column names specified by "from_name" and  "to_name" values.

### CoNLL2003

The CoNLL export feature generates a single output file, containing all available completions for all the tasks in the project. The resulting file has the following format:

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

Users can specify if only starred completions should be included in the output file by checking the _Only ground truth_ option before generating the export.

## Export Options

<img class="image image__shadow" src="/assets/images/annotation_lab/4.1.0/annotation_export.png" style="width:100%;"/>

**Tags**

Only allow export of tasks having the specified tags.

**Only Ground Truth**

If this option is enabled then only the tasks having ground truth in the completion will be exported.

**Exclude tasks without Completions**

Previous versions of the Annotation Lab only allowed the export of tasks that contained completions. From version <bl>2.8.0</bl> on, the tasks without any completions can be exported as this can be necessary for cloning projects. In the case where only tasks with completions are required in the export, users can enable the _Exclude tasks without Completions_ option on the Export page.