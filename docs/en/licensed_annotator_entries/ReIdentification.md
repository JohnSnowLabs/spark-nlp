{%- capture title -%}
ReIdentification
{%- endcapture -%}

{%- capture description -%}
Reidentifies obfuscated entities by DeIdentification. This annotator requires the outputs
from the deidentification as input. Input columns need to be the deidentified document and the deidentification
mappings set with DeIdentification.setMappingsColumn.
To see how the entities are deidentified, please refer to the example of that class.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT,CHUNK
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Define the reidentification stage and transform the deidentified documents
reideintification = ReIdentification() \
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

{%- capture scala_example -%}
// Define the reidentification stage and transform the deidentified documents
val reideintification = new ReIdentification()
  .setInputCols("dei", "protectedEntities")
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

{%- capture api_link -%}
[ReIdentification](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/deid/ReIdentification)
{%- endcapture -%}

{% include templates/licensed_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}