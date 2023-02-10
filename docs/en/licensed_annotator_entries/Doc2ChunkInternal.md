{%- capture title -%}
Doc2ChunkInternal
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

Converts `DOCUMENT`, `TOKEN` typed annotations into `CHUNK` type with the contents of a `chunkCol`. Chunk text must be contained within input `DOCUMENT`. May be either `StringType` or `ArrayType[StringType]` (using `setIsArray`). Useful for annotators that require a CHUNK type input.

For more extended examples on document pre-processing see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).


{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}

import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
tokenizer = Tokenizer().setInputCol("document").setOutputCol("token")
chunkAssembler = (
    Doc2ChunkInternal()
    .setInputCols("document", "token")
    .setChunkCol("target")
    .setOutputCol("chunk")
    .setIsArray(True)
)

data = spark.createDataFrame(
    [
        [
            "Spark NLP is an open-source text processing library for advanced natural language processing.",
            ["Spark NLP", "text processing library", "natural language processing"],
        ]
    ]
).toDF("text", "target")

pipeline = (
    Pipeline().setStages([documentAssembler, tokenizer, chunkAssembler]).fit(data)
)

result = pipeline.transform(data)
result.selectExpr("chunk.result", "chunk.annotatorType").show(truncate=False)
+-----------------------------------------------------------------+---------------------+
|result                                                           |annotatorType        |
+-----------------------------------------------------------------+---------------------+
|[Spark NLP, text processing library, natural language processing]|[chunk, chunk, chunk]|
+-----------------------------------------------------------------+---------------------+

{%- endcapture -%}


{%- capture model_api_link -%}
[Doc2ChunkInternal](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/annotator/Doc2ChunkInternal.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[Doc2ChunkInternal](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/doc2_chunk_internal/index.html#sparknlp_jsl.annotator.doc2_chunk_internal.Doc2ChunkInternal)
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
