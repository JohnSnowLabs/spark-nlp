{%- capture title -%} NomicEmbeddings {%- endcapture -%} 
{%- capture description -%} 
Nomic Embeddings are open-source text embedding models released by Nomic AI designed for semantic tasks such as search, clustering, classification, and retrieval. These models are optimized for high performance on the Massive Text Embedding Benchmark (MTEB) while remaining efficient and production-ready. Nomic provides different embedding sizes for balancing accuracy and compute efficiency, and the models are widely used in open-source vector databases like Atlas.  

They are trained on large, diverse datasets with careful curation to maximize generalization and multilingual coverage. 

Pretrained models can be loaded with the `pretrained` method of the companion object:
```scala
val embeddings = NomicEmbeddings.pretrained("nomic_embed_v1","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("embeddings")
```
The default model is `"nomic_embed_v1"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=NomicEmbeddings).

Spark NLP also supports Hugging Face transformer-based translation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resources**:
- [Nomic Embeddings on Hugging Face](https://huggingface.co/nomic-ai)  
- [Nomic AI Official Site](https://nomic.ai/)  
- [Nomic Atlas Vector Database](https://atlas.nomic.ai)  
- [Nomic Embed: Training a Reproducible Long Context Text Embedder (Arxiv)](https://arxiv.org/abs/2402.01613)  
- [Training Sparse Mixture Of Experts Text Embedding Models (Arxiv)](https://arxiv.org/abs/2502.07972)  
- [Nomic Embed Vision: Expanding the Latent Space (Arxiv)](https://arxiv.org/abs/2406.18587)  

**Paper abstract**

*This technical report describes the training of nomic-embed-text-v1, the first fully reproducible, open-source, open-weights, open-data, 8192 context length English text embedding model that outperforms both OpenAI Ada-002 and OpenAI text-embedding-3-small on the short-context MTEB benchmark and the long context LoCo benchmark. We release the training code and model weights under an Apache 2.0 license. In contrast with other open-source models, we release the full curated training data and code that allows for full replication of nomic-embed-text-v1. You can find code and data to replicate the model at [this https URL.](https://github.com/nomic-ai/contrastors)*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
EMBEDDINGS
{%- endcapture -%}

{%- capture api_link -%}
[NLLBTransformer](/api/com/johnsnowlabs/nlp/embeddings/NomicEmbeddings.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[NLLBTransformer](/api/python/reference/autosummary/sparknlp/annotator/embeddings/nomic_embeddings/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[NLLBTransformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/NomicEmbeddings.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import NomicEmbeddings
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")   

embeddings = NomicEmbeddings.pretrained("nomic_embed_v1","en") \
      .setInputCols(["document"]) \
      .setOutputCol("embeddings") 

pipeline = Pipeline().setStages([
    documentAssembler, 
    embeddings
])

data = spark.createDataFrame([
    ["Artificial intelligence is transforming the way people communicate, learn, and work across the world."]
]).toDF("text")

model = pipeline.fit(data)
result = model.transform(data)

result.select("embeddings.embeddings").show()

+--------------------+
|          embeddings|
+--------------------+
|[[-0.004530675, 0...|
+--------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.NomicEmbeddings
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = NomicEmbeddings.pretrained("nomic_embed_v1", "en")
  .setInputCols("document")
  .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings
))

val data = Seq("Artificial intelligence is transforming the way people communicate, learn, and work across the world.").toDF("text")

val model = pipeline.fit(data)
val result = model.transform(data)

result.select("embeddings.embeddings").show()

+--------------------+
|          embeddings|
+--------------------+
|[[-0.004530675, 0...|
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