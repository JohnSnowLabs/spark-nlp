{%- capture title -%}
ReIdentification
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}
Reidentifies obfuscated entities by DeIdentification. This annotator requires the outputs
from the deidentification as input. Input columns need to be the deidentified document and the deidentification
mappings set with DeIdentification.setMappingsColumn.
To see how the entities are deidentified, please refer to the example of that class.
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT,CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_python_medical -%}
from johnsnowlabs import * 

# Define the reidentification stage and transform the deidentified documents
reideintification = medical.ReIdentification() \
    .setInputCols(["dei", "protectedEntities"]) \
    .setOutputCol("reid") \
    .transform(result)

# Show results
result.select("dei.result").show(truncate = False)
+--------------------------------------------------------------------------------------------------+
|result                                                                                            |
+--------------------------------------------------------------------------------------------------+
|[# 01010101 Date : 01/18/93 PCP : Dr. Gregory House , <AGE> years-old , Record date : 2079-11-14.]|
+--------------------------------------------------------------------------------------------------+

reideintification.selectExpr("explode(reid.result)").show(truncate=False)
+-----------------------------------------------------------------------------------+
|col                                                                                |
+-----------------------------------------------------------------------------------+
|# 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09.|
+-----------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture model_scala_medical -%}
from johnsnowlabs import * 
// Define the reidentification stage and transform the deidentified documents
val reideintification = new medical.ReIdentification()
  .setInputCols(Array("dei", "protectedEntities"))
  .setOutputCol("reid")
  .transform(result)

// Show results
//
// result.select("dei.result").show(truncate = false)
// +--------------------------------------------------------------------------------------------------+
// |result                                                                                            |
// +--------------------------------------------------------------------------------------------------+
// |[# 01010101 Date : 01/18/93 PCP : Dr. Gregory House , <AGE> years-old , Record date : 2079-11-14.]|
// +--------------------------------------------------------------------------------------------------+
// reideintification.selectExpr("explode(reid.result)").show(false)
// +-----------------------------------------------------------------------------------+
// |col                                                                                |
// +-----------------------------------------------------------------------------------+
// |# 7194334 Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-11-09.|
// +-----------------------------------------------------------------------------------+
//
{%- endcapture -%}


{%- capture model_python_legal -%}
from johnsnowlabs import * 

# Define the reidentification stage and transform the deidentified documents
reideintification = legal.ReIdentification() \
    .setInputCols(["aux", "deidentified"]) \
    .setOutputCol("original") \
    .transform(result)

{%- endcapture -%}

{%- capture model_scala_legal -%}
from johnsnowlabs import * 
// Define the reidentification stage and transform the deidentified documents
val reideintification = new legal.ReIdentification()
  .setInputCols(Array("aux", "deidentified"))
  .setOutputCol("original")
  .transform(result)

{%- endcapture -%}

{%- capture model_python_finance -%}
from johnsnowlabs import * 

# Define the reidentification stage and transform the deidentified documents
reideintification = finance.ReIdentification() \
    .setInputCols(["aux", "deidentified"]) \
    .setOutputCol("original") \
    .transform(result)

{%- endcapture -%}

{%- capture model_scala_finance -%}
from johnsnowlabs import * 
// Define the reidentification stage and transform the deidentified documents
val reideintification = new finance.ReIdentification()
  .setInputCols(Array("aux", "deidentified"))
  .setOutputCol("original")
  .transform(result)

{%- endcapture -%}

{%- capture model_api_link -%}
[ReIdentification](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/ReIdentification)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_python_finance=model_python_finance
model_scala_finance=model_scala_finance
model_python_legal=model_python_legal
model_scala_legal=model_scala_legal
model_api_link=model_api_link%}


