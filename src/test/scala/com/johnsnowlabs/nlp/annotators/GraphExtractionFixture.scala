package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DEPENDENCY, LABELED_DEPENDENCY}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StructField, StructType}

trait GraphExtractionFixture {

  def getUniqueEntitiesDataSet(spark: SparkSession, tokenizerPipeline: Pipeline): DataFrame = {
    import spark.implicits._

    val textDataSet = Seq("United canceled the morning flights to Houston").toDS.toDF("text")
    val tokenDataSet = tokenizerPipeline.fit(textDataSet).transform(textDataSet)

    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 5, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14",
        "sentence" -> "0"), List()),
        Row(DEPENDENCY, 7, 14, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 16, 18, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 20, 26, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 28, 34, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 36, 37, "Houston", Map("head" -> "7", "head.begin" -> "39", "head.end" -> "45",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 39, 45, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 5, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 7, 14, "root", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 16, 18, "det", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 20, 26, "compound", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 28, 34, "obj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 36, 37, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 39, 45, "nmod", Map("sentence" -> "0"), List())
      )
    ))
    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))
    val mockDependencyParserDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockDependencyParserData),
      dependenciesStruct)

    val mockEntitiesData = Seq(Row(
      List(Row(CHUNK, 0, 5, "United", Map("entity" -> "ORG", "sentence" -> "0"), List()),
        Row(CHUNK, 20, 26, "morning", Map("entity" -> "TIME", "sentence" -> "0"), List()),
        Row(CHUNK, 39, 45, "Houston", Map("entity" -> "LOC", "sentence" -> "0"), List()))
    ))
    val entitiesStruct = mockStructType(List(("entities", CHUNK)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockEntitiesData), entitiesStruct)

    val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

    tokenDataSet.join(mockAnnotatorsDataSet)
  }

  def getAmbiguousEntitiesDataSet(spark: SparkSession, tokenizerPipeline: Pipeline): DataFrame = {
    import spark.implicits._

    val textDataSet = Seq("United canceled the morning flights to Houston and Dallas").toDS.toDF("text")
    val tokenDataSet = tokenizerPipeline.fit(textDataSet).transform(textDataSet)

    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 5, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14",
        "sentence" -> "0"), List()),
        Row(DEPENDENCY, 7, 14, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 16, 18, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 20, 26, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 28, 34, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 36, 37, "Houston", Map("head" -> "7", "head.begin" -> "39", "head.end" -> "45",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 39, 45, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 47, 49, "Dallas", Map("head" -> "9", "head.begin" -> "51", "head.end" -> "56",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 51, 56, "Houston", Map("head" -> "7", "head.begin" -> "39", "head.end" -> "45",
          "sentence" -> "0"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 5, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 7, 14, "root", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 16, 18, "det", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 20, 26, "compound", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 28, 34, "obj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 36, 37, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 39, 45, "nmod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 47, 49, "cc", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 51, 56, "conj", Map("sentence" -> "0"), List())
      )
    ))

    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))
    val mockDependencyParserDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockDependencyParserData),
      dependenciesStruct)

    val mockEntitiesData = Seq(Row(
      List(Row(CHUNK, 0, 5, "United", Map("entity" -> "ORG", "sentence" -> "0"), List()),
        Row(CHUNK, 20, 26, "morning", Map("entity" -> "TIME", "sentence" -> "0"), List()),
        Row(CHUNK, 39, 45, "Houston", Map("entity" -> "LOC", "sentence" -> "0"), List()),
        Row(CHUNK, 51, 56, "Dallas", Map("entity" -> "LOC", "sentence" -> "0"), List())
       )
    ))
    val entitiesStruct = mockStructType(List(("entities", CHUNK)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockEntitiesData), entitiesStruct)

    val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

    tokenDataSet.join(mockAnnotatorsDataSet)
  }

  def getEntitiesFromTwoSentences(spark: SparkSession, pipeline: Pipeline): DataFrame = {
    import spark.implicits._

    val textDataSet = Seq("United canceled the morning flights to Houston. So, I will go to London tomorrow")
      .toDS.toDF("text")
    val tokenDataSet = pipeline.fit(textDataSet).transform(textDataSet)

    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 5, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14",
        "sentence" -> "0"), List()),
        Row(DEPENDENCY, 7, 14, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 16, 18, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 20, 26, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 28, 34, "canceled", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 36, 37, "Houston", Map("head" -> "7", "head.begin" -> "39", "head.end" -> "45",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 39, 45, "flights", Map("head" -> "5", "head.begin" -> "28", "head.end" -> "34",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 46, 46, ".", Map("head" -> "2", "head.begin" -> "7", "head.end" -> "14",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 48, 49, "go" , Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 50, 50, "go" , Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 52, 52, "go" , Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 54, 57, "go" , Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 59, 60, "root" , Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 62, 63, "go" , Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 65, 70, "go" , Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 72, 79, "go" , Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 5, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 7, 14, "root", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 16, 18, "det", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 20, 26, "compound", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 28, 34, "obj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 36, 37, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 39, 45, "nmod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 46, 46, "punct", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 48, 49, "advmod", Map("sentence" -> "1"), List()),
        Row(LABELED_DEPENDENCY, 50, 50, "punct", Map("sentence" -> "1"), List()),
        Row(LABELED_DEPENDENCY, 52, 52, "nsubj", Map("sentence" -> "1"), List()),
        Row(LABELED_DEPENDENCY, 54, 57, "aux", Map("sentence" -> "1"), List()),
        Row(LABELED_DEPENDENCY, 59, 60, "root", Map("sentence" -> "1"), List()),
        Row(LABELED_DEPENDENCY, 62, 63, "case", Map("sentence" -> "1"), List()),
        Row(LABELED_DEPENDENCY, 65, 70, "obl", Map("sentence" -> "1"), List()),
        Row(LABELED_DEPENDENCY, 72, 79, "obl:tmod", Map("sentence" -> "1"), List())
      )
    ))
    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))
    val mockDependencyParserDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockDependencyParserData),
      dependenciesStruct)

    val mockEntitiesData = Seq(Row(
      List(Row(CHUNK, 0, 5, "United", Map("entity" -> "ORG", "sentence" -> "0"), List()),
        Row(CHUNK, 20, 26, "morning", Map("entity" -> "TIME", "sentence" -> "0"), List()),
        Row(CHUNK, 39, 45, "Houston", Map("entity" -> "LOC", "sentence" -> "0"), List()),
        Row(CHUNK, 65, 70, "London", Map("entity" -> "LOC", "sentence" -> "1"), List()),
        Row(CHUNK, 72, 79, "tomorrow", Map("entity" -> "TIME", "sentence" -> "1"), List())
      )
    ))
    val entitiesStruct = mockStructType(List(("entities", CHUNK)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockEntitiesData), entitiesStruct)

    val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

    tokenDataSet.join(mockAnnotatorsDataSet)
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
