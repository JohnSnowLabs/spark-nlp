package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.size
import org.scalatest._


class XlnetEmbeddingsTestSpec extends FlatSpec {

  "Xlnet Embeddings" should "correctly load pretrained model" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = XlnetEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    println(pipelineDF.count())
    pipelineDF.show()
    //    pipelineDF.printSchema()
    pipelineDF.select("token.result").show(4, false)
    pipelineDF.select("embeddings.result").show(4, false)
    pipelineDF.select("embeddings.metadata").show(4, false)
    pipelineDF.select("embeddings.embeddings").show(4, truncate = 300)
    pipelineDF.select(size(pipelineDF("embeddings.embeddings")).as("embeddings_size")).show
    Benchmark.time("Time to save XlnetEmbeddings results") {
      pipelineDF.select("embeddings").write.mode("overwrite").parquet("./tmp_xlnet_embeddings")
    }

  }

}
