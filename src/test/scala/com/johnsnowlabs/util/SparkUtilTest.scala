package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{SparkSessionTest, Tokenizer}
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.spark.SparkUtil
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class SparkUtilTest extends AnyFlatSpec with SparkSessionTest {

  "SparkUtil" should "retrieve column name for Token annotator type " taggedAs FastTest in {
    val expectedColumn = "token"
    val testDataset = tokenizerPipeline.fit(emptyDataSet).transform(emptyDataSet)

    val actualColumn = SparkUtil.retrieveColumnName(testDataset, TOKEN)

    assert(expectedColumn == actualColumn)
  }

  it should "retrieve custom column name for Token annotator type " taggedAs FastTest in {
    val customColumnName = "my_custom_token_col"
    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol(customColumnName)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))
    val testDataset = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    val actualColumn = SparkUtil.retrieveColumnName(testDataset, TOKEN)

    assert(customColumnName == actualColumn)
  }

}
