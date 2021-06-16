package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DEPENDENCY, LABELED_DEPENDENCY}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types._
import org.scalatest.FlatSpec

class GraphExtractorTest extends FlatSpec with SparkSessionTest {

  import spark.implicits._

  private val mockDependencyParserDataSet = getMockDependencyParserDataSet
  private val mockEntitiesDataSet = getMockEntityDataSet
  private val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

  "Graph Extractor" should "return dependency graphs between entities" in {

    val textDataSet = Seq("United canceled the morning flights to Houston").toDS.toDF("text")
    val tokenDataSet = tokenizerPipeline.fit(textDataSet).transform(textDataSet)
    val testDataSet = tokenDataSet.join(mockAnnotatorsDataSet)
    testDataSet.show(false)

    val graphExtractor = new GraphExtractor()
      .setInputCols("token", "heads", "deprel", "entities")
      .setOutputCol("graph")

    val result = graphExtractor.transform(testDataSet)

    result.select("graph").show(false)

  }

  private def getMockDependencyParserDataSet: DataFrame = {
    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 5, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14"), List()),
        Row(DEPENDENCY, 7, 14, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1"), List()),
        Row(DEPENDENCY, 16, 18, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34"), List()),
        Row(DEPENDENCY, 20, 26, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34"), List()),
        Row(DEPENDENCY, 28, 34, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14"), List()),
        Row(DEPENDENCY, 36, 37, "Houston", Map("head" -> "7", "head.begin" -> "39", "head.end" -> "45"), List()),
        Row(DEPENDENCY, 39, 45, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 5, "nsubj", Map(), List()),
        Row(LABELED_DEPENDENCY, 7, 14, "root", Map(), List()),
        Row(LABELED_DEPENDENCY, 16, 18, "det", Map(), List()),
        Row(LABELED_DEPENDENCY, 20, 26, "compound", Map(), List()),
        Row(LABELED_DEPENDENCY, 28, 34, "obj", Map(), List()),
        Row(LABELED_DEPENDENCY, 36, 37, "case", Map(), List()),
        Row(LABELED_DEPENDENCY, 39, 45, "nmod", Map(), List())
      )
    ))
    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))

    spark.createDataFrame(sparkContext.parallelize(mockDependencyParserData), dependenciesStruct)
  }

  private def getMockEntityDataSet: DataFrame = {
    val mockEntitiesData = Seq(Row(
      List(Row(CHUNK, 0, 5, "United", Map("entity" -> "ORG"), List()),
        Row(CHUNK, 20, 26, "morning", Map("entity" -> "TIME"), List()),
        Row(CHUNK, 39, 45, "Houston", Map("entity" -> "LOC"), List()))
    ))

    val entitiesStruct = mockStructType(List(("entities", CHUNK)))

    spark.createDataFrame(sparkContext.parallelize(mockEntitiesData), entitiesStruct)
  }

  private def mockStructType(columnsAndAnnotators: List[(String, String)]): StructType = {
    val structFields: List[StructField] = columnsAndAnnotators.map{ columnAndAnnotator =>
      val columnName = columnAndAnnotator._1
      val annotatorType = columnAndAnnotator._2
      val metadataBuilder: MetadataBuilder = new MetadataBuilder()
      metadataBuilder.putString("annotatorType", annotatorType)
      val outputField = StructField(columnName, ArrayType(Annotation.dataType), nullable = false, metadataBuilder.build)
      outputField
    }
    StructType(structFields)
  }

}
