package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec

class DocumentTokenSplitterTest extends AnyFlatSpec {

  val spark = ResourceHelper.spark

  import spark.implicits._

  val text =
    "All emotions, and that\none particularly, were abhorrent to his cold, precise but\nadmirably balanced mind.\n\n" +
      "He was, I take it, the most perfect\nreasoning and observing machine that the world has seen."

  val textDf = Seq(text).toDF("text")
  val documentAssembler = new DocumentAssembler().setInputCol("text")
  val textDocument: DataFrame = documentAssembler.transform(textDf)

  behavior of "DocumentTokenTextSplitter"

  it should "split by number of tokens" taggedAs FastTest in {
    val numTokens = 3
    val tokenTextSplitter =
      new DocumentTokenSplitter()
        .setInputCols("document")
        .setOutputCol("splits")
        .setNumTokens(numTokens)

    val splitDF = tokenTextSplitter.transform(textDocument)
    val result = Annotation.collect(splitDF, "splits").head

    result.foreach(annotation => assert(annotation.metadata("numTokens").toInt == numTokens))
  }

  it should "split tokens with overlap" taggedAs FastTest in {
    val numTokens = 3
    val tokenTextSplitter =
      new DocumentTokenSplitter()
        .setInputCols("document")
        .setOutputCol("splits")
        .setNumTokens(numTokens)
        .setTokenOverlap(1)

    val splitDF = tokenTextSplitter.transform(textDocument)
    val result = Annotation.collect(splitDF, "splits").head

    result.zipWithIndex.foreach { case (annotation, i) =>
      if (i < result.length - 1) // Last document is shorter
        assert(annotation.metadata("numTokens").toInt == numTokens)
    }
  }

}
