package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructField
import org.scalatest.flatspec.AnyFlatSpec

class CoNLLTestSpec extends AnyFlatSpec {

  private val expectedColumnNames: Seq[String] =
    Seq("document", "sentence", "token", "pos", "label")
  private val expectedAnnotatorTypes: Seq[String] =
    Seq("document", "document", "token", "pos", "named_entity")

  private def assertNamesAndTypes(trainingDataSchema: Seq[StructField]) = {
    val annotatorSchema = trainingDataSchema
    val comparedColumnNames = annotatorSchema.map(_.name).zip(expectedColumnNames)
    val comparedAnnotationTypes =
      annotatorSchema.map(_.metadata.getString("annotatorType")).zip(expectedAnnotatorTypes)
    assert((comparedColumnNames ++ comparedAnnotationTypes).forall(x => x._1 == x._2))
  }

  private def assertDocIds(trainingData: Dataset[_], expectedDocIds: Seq[String]) = {
    val comparedDocIds =
      trainingData.select("doc_id").collect.map(_.getString(0)).zip(expectedDocIds).toSeq
    assert(comparedDocIds.forall(x => x._1 == x._2))
  }

  "CoNLL" should "read a CoNLL and have the columns in the right order" taggedAs FastTest in {
    val trainingData =
      CoNLL().readDataset(ResourceHelper.spark, "src/test/resources/conll/test_conll_docid.txt")
    assertNamesAndTypes(trainingData.schema.tail)
  }

  "CoNLL" should "read a CoNLL and have the columns in the right order using * pattern" taggedAs FastTest in {
    val trainingData = CoNLL().readDataset(ResourceHelper.spark, "src/test/resources/conll/*")
    assertNamesAndTypes(trainingData.schema.tail)
  }

  "CoNLL" should "read a CoNLL with ids and have the columns in the right order" taggedAs FastTest in {
    val trainingData = CoNLL(includeDocId = true)
      .readDataset(ResourceHelper.spark, "src/test/resources/conll/test_conll_docid.txt")
    val expectedDocIds = Seq("O", "1", "2", "3-1", "3-2")
    assertNamesAndTypes(trainingData.schema.tail.tail)
    assertDocIds(trainingData, expectedDocIds)
  }

  "CoNLL" should "read a CoNLL with ids and have the columns in the right order using * pattern" taggedAs FastTest in {
    val trainingData =
      CoNLL(includeDocId = true).readDataset(ResourceHelper.spark, "src/test/resources/conll/*")
    val expectedDocIds = Seq("O", "1", "2", "3-1", "3-2")
    assertNamesAndTypes(trainingData.schema.tail.tail)
    assertDocIds(trainingData, expectedDocIds)
  }

}
