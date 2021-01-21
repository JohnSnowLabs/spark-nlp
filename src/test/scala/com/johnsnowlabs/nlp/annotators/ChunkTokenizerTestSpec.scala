package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

class ChunkTokenizerTestSpec extends FlatSpec {

  "a ChunkTokenizer" should "correctly identify origin source and in correct order" taggedAs FastTest in {

    import SparkAccessor.spark.implicits._

    val data = Seq(
      "Hello world, my name is Michael, I am an artist and I work at Benezar",
      "Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."
    ).toDS.toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val entityExtractor = new TextMatcher()
      .setInputCols("sentence", "token")
      .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.TEXT)
      .setOutputCol("entity")

    val chunkTokenizer = new ChunkTokenizer()
      .setInputCols("entity")
      .setOutputCol("chunk_token")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        entityExtractor,
        chunkTokenizer
      ))

    val result = pipeline.fit(data).transform(data)

    result.select("entity", "chunk_token").as[(Array[Annotation], Array[Annotation])].foreach(column => {
      val chunks = column._1
      val chunkTokens = column._2
      chunkTokens.foreach{chunkToken => {
        val index = chunkToken.metadata("chunk").toInt
        require(chunks.apply(index).result.contains(chunkToken.result), s"because ${chunks(index)} does not contain ${chunkToken.result}")
      }}
      require(chunkTokens.flatMap(_.metadata.values).distinct.length == chunks.length, s"because amount of chunks ${chunks.length} does not equal to amount of token belongers")
    })

    succeed

  }

}
