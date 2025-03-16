{%- capture title -%}
OLMoTransformer
{%- endcapture -%}

{%- capture description -%}
OLMo, a series of Open Language Models, is designed to enable the science of language models. These models are trained on the Dolma dataset, offering open-source capabilities for language model research and application. The OLMo models support various NLP tasks including text generation, summarization, and more.

Pretrained models can be loaded using the `pretrained` method from the companion object:


```scala
val olmo = OLMoTransformer.pretrained()
  .setInputCols("document")
  .setOutputCol("generation")
```

The default model is `"olmo_1b_int4"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=OLMo).

For extended examples of usage, see
[OLMoTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/OLMoTestSpec.scala).

**Sources** :
[OLMo Project Page](https://allenai.org/olmo)
[OLMo GitHub Repository](https://github.com/allenai/OLMo)
[OLMo: Accelerating the Science of Language Models (Paper)](https://arxiv.org/pdf/2402.00838.pdf)

**Paper abstract**

*Language models (LMs) have become ubiquitous in both NLP research and commercial products. 
As their commercial importance has surged, the most powerful models have become proprietary, 
limiting scientific study. OLMo addresses this gap by offering an open-source framework, 
including training data, models, and code. This initiative aims to empower the research community, 
fostering transparency and innovation in language model development.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Document Assembler
document_assembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

# OLMo Transformer
olmo = OLMoTransformer.pretrained("olmo_1b_int4") \
.setInputCols(["document"]) \
.setMinOutputLength(10) \
.setMaxOutputLength(50) \
.setDoSample(False) \
.setTopK(50) \
.setNoRepeatNgramSize(3) \
.setOutputCol("generation")

# Pipeline
pipeline = Pipeline(stages=[document_assembler, olmo])

# Sample Data
data = spark.createDataFrame([["My name is Leonardo."]]).toDF("text")
result = pipeline.fit(data).transform(data)

# Display Results
result.select("generation.result").show(truncate=False)

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.seq2seq.OLMoTransformer
import org.apache.spark.ml.Pipeline

// Document Assembler
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

// OLMo Transformer
val olmo = OLMoTransformer.pretrained("olmo_1b_int4")
.setInputCols(Array("document"))
.setMinOutputLength(10)
.setMaxOutputLength(50)
.setDoSample(false)
.setTopK(50)
.setNoRepeatNgramSize(3)
.setOutputCol("generation")

// Pipeline
val pipeline = new Pipeline().setStages(Array(documentAssembler, olmo))

// Sample Data
val data = Seq("My name is Leonardo.").toDF("text")
val result = pipeline.fit(data).transform(data)

// Display Results
result.select("generation.result").show(truncate = false)

{%- endcapture -%}

{%- capture api_link -%}
[OLMoTransformer](/api/com/johnsnowlabs/nlp/seq2seq/OLMoTransformer)
{%- endcapture -%}

{%- capture python_api_link -%}
[OLMoTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/olmo_transformer/index.html#sparknlp.annotator.seq2seq.olmo_transformer.OLMoTransformer)
{%- endcapture -%}

{%- capture source_link -%}
[OLMoTransformer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/seq2seq/OLMoTransformer.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}