{%- capture title -%}
ChunkMapperFilterer
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

`ChunkMapperFilterer` is an annotator to be used after `ChunkMapper` that alloows to filter chunks based on the results of the mapping, whether it was successful or failed.

Example usage and more details can be found on Spark NLP Workshop repository accessible in GitHub, for example the notebook [Healthcare Chunk Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb).

{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK, LABEL_DEPENDENCY
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}


{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkMapperFilterer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/chunker/ChunkMapperFilterer.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ChunkMapperFilterer](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/chunker/chunkmapper_filterer/index.html#sparknlp_jsl.annotator.chunker.chunkmapper_filterer.ChunkMapperFilterer)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
approach=approach
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
