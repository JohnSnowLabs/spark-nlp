{%- capture title -%}
DocumentHashCoder
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

This annotator can replace dates in a column of `DOCUMENT` type according with the hash code of any other column. It uses the hash of the specified column and creates a new document column containing the day shift information. In sequence, the `DeIdentification` annotator deidentifies the document with the shifted date information. 

If the specified column contains strings that can be parsed to integers, use those numbers to make the shift in the data accordingly.

{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_python_medical -%}

import pandas as pd


data = pd.DataFrame(
    {'patientID' : ['A001', 'A001', 
                    'A003', 'A003'],
     'text' : ['Chris Brown was discharged on 10/02/2022', 
               'Mark White was discharged on 10/04/2022', 
               'John was discharged on 15/03/2022',
               'John Moore was discharged on 15/12/2022'
              ],
     'dateshift' : ['10', '10', 
                    '30', '30']
    }
)

my_input_df = spark.createDataFrame(data)

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

documentHasher = DocumentHashCoder()\
    .setInputCols("document")\
    .setOutputCol("document2")\
    .setDateShiftColumn("dateshift")

tokenizer = Tokenizer()\
    .setInputCols(["document2"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["document2", "token"])\
    .setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel\
    .pretrained("ner_deid_subentity_augmented", "en", "clinical/models")\
    .setInputCols(["document2","token", "word_embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
    .setInputCols(["document2", "token", "ner"])\
    .setOutputCol("ner_chunk")

de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "document2"]) \
    .setOutputCol("deid_text") \
    .setMode("obfuscate") \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setLanguage("en") \
    .setObfuscateRefSource('faker') \
    .setUseShifDays(True)

pipeline_col = Pipeline().setStages([
    documentAssembler,
    documentHasher,
    tokenizer,
    embeddings,
    clinical_ner,
    ner_converter,
    de_identification
])

empty_data = spark.createDataFrame([["", "", ""]]).toDF("patientID","text", "dateshift")
pipeline_col_model = pipeline_col.fit(empty_data)

output = pipeline_col_model.transform(my_input_df)
output.select('text', 'dateshift', 'deid_text.result').show(truncate = False)

+----------------------------------------+---------+----------------------------------------------+
text                                    |dateshift|result                                        |
+----------------------------------------+---------+----------------------------------------------+
Chris Brown was discharged on 10/02/2022|10       |[Ellender Manual was discharged on 20/02/2022]|
Mark White was discharged on 10/04/2022 |10       |[Errol Bang was discharged on 20/04/2022]     |
John was discharged on 15/03/2022       |30       |[Ariel Null was discharged on 14/04/2022]     |
John Moore was discharged on 15/12/2022 |30       |[Jean Cotton was discharged on 14/01/2023]    |
+----------------------------------------+---------+----------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[DocumentHashCoder](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/DocumentHashCoder.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[DocumentHashCoder](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/deid/doccument_hashcoder/index.html#sparknlp_jsl.annotator.deid.doccument_hashcoder.DocumentHashCoder)
{%- endcapture -%}

{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
