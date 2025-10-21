{%- capture title -%} MxbaiEmbeddings {%- endcapture -%} 
{%- capture description -%} 
Mxbai Embeddings are a family of models by Mixedbread AI that convert text into vector representations for tasks like semantic search, document retrieval, clustering, classification, and recommendation. They feature binary quantization (up to ~32× smaller with minimal performance loss), Matryoshka Representation Learning (MRL) (front-loaded dimensions for efficient truncation), long context support (up to 4096 tokens in the xsmall model), are open-source under Apache-2.0, and achieve strong MTEB benchmark scores (~64.68 for the large model).

Pretrained models can be loaded with the `pretrained` method of the companion object:
```scala
val mxbai = MxbaiEmbeddings.pretrained("mxbai_large_v1", "en")
    .setInputCols("documents")
    .setOutputCol("embeddings")
```
The default model is `"mxbai_large_v1"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=MxbaiEmbeddings).  

Spark NLP supports a variety of Hugging Face embedding models. To learn how to import and use them, check out the following thread:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [Mxbai Models on Hugging Face](https://huggingface.co/mixedbread-ai)  
- [Mixedbread Docs — Our Embedding Models](https://www.mixedbread.com/docs/models/embedding)  
- [GitHub: binary-embeddings (quantization showcase)](https://github.com/mixedbread-ai/binary-embeddings)

**Paper abstract**

*We present Mxbai Embeddings, a family of open-source text embedding models developed by Mixedbread AI for high-performance semantic representation learning. The flagship model, mxbai-embed-large-v1, is trained on over 700 million text pairs and fine-tuned with 30 million high-quality triplets using the AnglE loss function. These models are designed for a broad range of applications including semantic search, retrieval, clustering, classification, and recommendation. Key innovations include binary quantization, which reduces embedding storage size by up to 32× and accelerates retrieval speeds with minimal accuracy loss, and Matryoshka Representation Learning (MRL), which prioritizes information in earlier embedding dimensions to enable efficient truncation. Selected models also support extended context lengths (up to 4096 tokens). All models are released under the Apache-2.0 license, ensuring broad usability. On the Massive Text Embedding Benchmark (MTEB), mxbai-embed-large-v1 achieves an average score of ~64.68, demonstrating competitive performance compared to proprietary systems while offering efficiency and transparency for the community.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENTS
{%- endcapture -%}

{%- capture output_anno -%}
EMBEDDINGS
{%- endcapture -%}

{%- capture api_link -%}
[MxbaiEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/MxbaiEmbeddings.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[MxbaiEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/mxbai_embeddings/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[MxbaiEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/MxbaiEmbeddings.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import MxbaiEmbeddings
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

mxbai = MxbaiEmbeddings.pretrained("mxbai_large_v1","en") \
    .setInputCols("document") \
    .setOutputCol("embeddings") \

pipeline = Pipeline().setStages([
    documentAssembler, 
    mxbai
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
|[[-0.26401705, 0....|
+--------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.MxbaiEmbeddings
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val mxbai = MxbaiEmbeddings.pretrained("mxbai_large_v1", "en")
  .setInputCols("document")
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  mxbai
))

val data = Seq("I love spark-nlp").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("embeddings.embeddings").show()

+--------------------+
|          embeddings|
+--------------------+
|[[-0.26401705, 0....|
+--------------------+
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