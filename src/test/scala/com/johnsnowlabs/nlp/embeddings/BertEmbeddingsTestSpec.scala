package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.scalatest._

class BertEmbeddingsTestSpec extends FlatSpec {

  "Bert Embeddings" should "correctly embed tokens and sentences" ignore {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Something is weird on the notebooks, something is happening."
    ).toDF("text")

    val data1 = Seq(
      "In the Seven Kingdoms of Westeros, a soldier of the ancient Night's Watch order survives an attack by supernatural creatures known as the White Walkers, thought until now to be mythical."
    ).toDF("text")

    val data2 = Seq(
      "In King's Landing, the capital, Jon Arryn, the King's Hand, dies under mysterious circumstances."
    ).toDF("text")

    val data3 = Seq(
      "Tyrion makes saddle modifications for Bran that will allow the paraplegic boy to ride."
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings.pretrained("bert_base_cased", "en")
      .setInputCols(Array("token", "document"))
      .setOutputCol("bert")

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, embeddings))

    val bertDDD = pipeline.fit(ddd).transform(ddd)
    val bertDF1 = pipeline.fit(data1).transform(data1)
    val bertDF2 = pipeline.fit(data2).transform(data2)
    val bertDF3 = pipeline.fit(data3).transform(data3)

  }

  "Bert Embeddings" should "correctly work in a pipeline" ignore {

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings
      ))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }

  }

  "Bert Embeddings" should "correctly work with empty tokens" ignore {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val embeddings = BertEmbeddings.pretrained()
      .setInputCols("document", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        stopWordsCleaner,
        embeddings
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    Benchmark.time("Time to save BertEmbeddings results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_bert_embeddings")
    }

  }

}
