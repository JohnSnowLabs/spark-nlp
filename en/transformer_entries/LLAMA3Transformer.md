{%- capture title -%}
LLAMA3Transformer
{%- endcapture -%}

{%- capture description -%}
[Llama 3](https://www.llama.com/models/llama-3/) is the next generation of Meta's large language models, available in 8B and 70B parameter sizes. Llama 3 introduces improvements in model architecture, training data scale, and context length, resulting in enhanced reasoning, code generation, and instruction-following capabilities compared to Llama 2.

Llama 3-Chat is fine-tuned for dialogue and assistant use cases, leveraging supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align outputs with human preferences.

Pretrained models can be loaded with `pretrained` of the companion object:
```scala
val seq2seq = LLAMA3Transformer.pretrained("llama_3_7b_instruct_hf_int4","en") 
    .setInputCols(Array("document")) 
    .setOutputCol("generation")
```
The default model is `"llama_3_7b_instruct_hf_int4"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=LLAMA3Transformer).

Spark NLP also supports Hugging Face transformer-based code generation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Resource**:

- [Meta Llama 3 on HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [Meta AI Llama 3 Announcement](https://ai.meta.com/blog/meta-llama-3/)
- [Meta Llama 3 Technical Report (2024)](https://arxiv.org/abs/2407.21783)
- [Awesome Llama Resources (GitHub)](https://github.com/MIBlue119/awesome-llama-resources#llama-3)
- [Fine-Tuning Llama 3: Guide (DataCamp)](https://www.datacamp.com/tutorial/llama3-fine-tuning-locally)

**Paper abstract**

*Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens. This paper presents an extensive empirical evaluation of Llama 3. We find that Llama 3 delivers comparable quality to leading language models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including pre-trained and post-trained versions of the 405B parameter language model and our Llama Guard 3 model for input and output safety. The paper also presents the results of experiments in which we integrate image, video, and speech capabilities into Llama 3 via a compositional approach. We observe this approach performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The resulting models are not yet being broadly released as they are still under development.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[LLAMA3Transformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/LLAMA3Transformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[LLAMA3Transformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/llama3_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[LLAMA3Transformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/LLAMA3Transformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import LLAMA3Transformer
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

llama3 = LLAMA3Transformer.pretrained() \
    .setInputCols(["documents"]) \
    .setOutputCol("generation") \
    .setMinOutputLength(50) \
    .setMaxOutputLength(250) \
    .setDoSample(True) \
    .setTemperature(0.7) \
    .setTopK(50) \
    .setTopP(0.9) \
    .setRepetitionPenalty(1.1) \
    .setNoRepeatNgramSize(3) \
    .setIgnoreTokenIds([])

pipeline = Pipeline().setStages([
    document_assembler,
    llama3
])

prompt = spark.createDataFrame([("""
### System:
You are a concise assistant who explains machine learning in simple terms.

### User:
Explain the difference between supervised and unsupervised learning with examples.

### Assistant:
""",)], ["text"])

model = pipeline.fit(prompt)
results = model.transform(prompt)

results.select("generation.result").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val llama3 = LLAMA3Transformer.pretrained()
  .setInputCols("documents")
  .setOutputCol("generation")
  .setMinOutputLength(50)
  .setMaxOutputLength(250)
  .setDoSample(true)
  .setTemperature(0.7f)
  .setTopK(50)
  .setTopP(0.9f)
  .setRepetitionPenalty(1.1f)
  .setNoRepeatNgramSize(3)
  .setIgnoreTokenIds(Array.emptyIntArray)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  llama3
))

val prompt = Seq("""
### System:
You are a concise assistant who explains machine learning in simple terms.

### User:
Explain the difference between supervised and unsupervised learning with examples.

### Assistant:
""").toDF("text")

val model = pipeline.fit(prompt)
val results = model.transform(prompt)

results.select("generation.result").show(false)
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