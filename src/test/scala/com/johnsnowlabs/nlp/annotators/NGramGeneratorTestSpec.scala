package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

class NGramGeneratorTestSpec extends FlatSpec {

  "NGramGenerator" should "correctly generate n-grams from tokenizer's results" in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. This is my forth.")
    )).toDF("id", "text")

    val expectedNGrams = Seq(
      Annotation(CHUNK, 0, 6, "This is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my first", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "first sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence .", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 33, "This is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 43, "my second", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 44, "second .", Map("sentence" -> "1")),
      Annotation(CHUNK, 0, 6, "This is", Map("sentence" -> "0")),
      Annotation(CHUNK, 5, 9, "is my", Map("sentence" -> "0")),
      Annotation(CHUNK, 8, 15, "my third", Map("sentence" -> "0")),
      Annotation(CHUNK, 11, 24, "third sentence", Map("sentence" -> "0")),
      Annotation(CHUNK, 17, 25, "sentence .", Map("sentence" -> "0")),
      Annotation(CHUNK, 27, 33, "This is", Map("sentence" -> "1")),
      Annotation(CHUNK, 32, 36, "is my", Map("sentence" -> "1")),
      Annotation(CHUNK, 35, 42, "my forth", Map("sentence" -> "1")),
      Annotation(CHUNK, 38, 43, "forth .", Map("sentence" -> "1"))
    )

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("ngrams")
      .setN(2)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        nGrams
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)
//    pipelineDF.show()
    pipelineDF.select("token").show(false)
    pipelineDF.select("ngrams").show(false)

    val nGramGeneratorResults = Annotation.collect(pipelineDF, "ngrams").flatten.toSeq

    assert(nGramGeneratorResults == expectedNGrams)

  }
}
