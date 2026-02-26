{%- capture title -%} SnowFlakeEmbeddings {%- endcapture -%} 
{%- capture description -%} 
Snowflake Embeddings are vector representations of text and images that let you measure *semantic similarity*, powering tasks like search, clustering, classification, and retrieval-augmented generation (RAG).  
The main family, called Arctic-embed, comes in different sizes for a balance of speed, cost, and accuracy, and supports *multilingual* and *long-context* inputs.  

Pretrained models can be loaded with the `pretrained` method of the companion object:

```scala
val snowflake = SnowFlakeEmbeddings.pretrained("snowflake_artic_m", "en")
    .setInputCols("documents")
    .setOutputCol("embeddings")
```

The default model is `"snowflake_artic_m"`, if no name is provided. For available pretrained models please see the Models Hub. 

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=SnowFlakeEmbeddings).

Spark NLP also supports Hugging Face transformer-based translation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [Snowflake Arctic Embed announcement blog](https://www.snowflake.com/en/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/)  
- [Snowflake-Labs / arctic-embed (GitHub)](https://github.com/Snowflake-Labs/arctic-embed)  
- [Snowflake on Hugging Face (organization)](https://huggingface.co/Snowflake)  

**Paper abstract**

*This report describes the training dataset creation and recipe behind the family of arctic-embed text embedding models (a set of five models ranging from 22 to 334 million parameters with weights open-sourced under an Apache-2 license). At the time of their release, each model achieved state-of-the-art retrieval accuracy for models of their size on the MTEB Retrieval leaderboard,1 with the largest model, arctic-embed-l outperforming closed source embedding models such as Cohere’s embed-v3 and Open AI’s text-embed-3-large. In addition to the details of our training recipe, we have provided several informative ablation studies, which we believe are the cause of our model performance.*

{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
EMBEDDINGS
{%- endcapture -%}

{%- capture api_link -%}
[SnowFlakeEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/SnowFlakeEmbeddings.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[SnowFlakeEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/snowflake_embeddings/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[SnowFlakeEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/SnowFlakeEmbeddings.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SnowFlakeEmbeddings
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
  .setInputCol("text") \
  .setOutputCol("document")   

snowflake = SnowFlakeEmbeddings.pretrained("snowflake_artic_m","en") \
    .setInputCols("document") \
    .setOutputCol("embeddings") \

pipeline = Pipeline().setStages([
    documentAssembler,
    snowflake
])

data = spark.createDataFrame([
    ["I love spark-nlp"]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("embeddings.embeddings").show()

+--------------------+
|          embeddings|
+--------------------+
|[[-0.6112396, 0.2...|
+--------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.SnowflakeEmbeddings
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val snowflake = SnowflakeEmbeddings.pretrained("snowflake_artic_m", "en")
  .setInputCols("document")
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  snowflake
))

val data = spark.createDataFrame(Seq(
  Tuple1("I love spark-nlp")
)).toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("embeddings.embeddings").show()

+--------------------+
|          embeddings|
+--------------------+
|[[-0.6112396, 0.2...|
+--------------------+
{%- endcapture -%}

{%- capture resources -%}
Resources:
- [Qwen official site (Qwen Chat)](https://qwen.ai/)  
- [Qwen GitHub (Qwen / Qwen)](https://github.com/QwenLM/Qwen)  
- [Qwen on Hugging Face (organization)](https://huggingface.co/Qwen)  
- [Qwen technical report / papers (arXiv)](https://arxiv.org/search/?query=Qwen&searchtype=all)  
- [Spark NLP QwenTransformer docs](https://sparknlp.org/api/python/reference/autosummary/sparknlp/annotator/seq2seq/qwen_transformer/index.html)  
{%- endcapture -%}

{%- capture paper_abstract -%}
*We introduce Qwen, a comprehensive language model series consisting of base and chat models across multiple sizes. The series demonstrates strong performance on a wide range of downstream tasks, with chat variants aligned via human feedback and specialized models for coding and math. Models support extended context lengths (commonly up to 32K tokens) and show competitive results compared to contemporaneous open-source models.*  
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
