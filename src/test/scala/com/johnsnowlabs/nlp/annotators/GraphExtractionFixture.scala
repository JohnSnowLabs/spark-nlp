package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{DEPENDENCY, LABELED_DEPENDENCY, NAMED_ENTITY}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

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

    val mockNerData = Seq(Row(
      List(Row(NAMED_ENTITY, 0, 5, "B-ORG", Map("entity" -> "United", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 7, 14, "O", Map("entity" -> "canceled", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 16, 18, "O", Map("entity" -> "the", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 20, 26, "B-TIME", Map("entity" -> "morning", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 28, 34, "O", Map("entity" -> "flights", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 36, 37, "O", Map("entity" -> "to", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 39, 45, "B-LOC", Map("entity" -> "Houston", "sentence" -> "0"), List()))
    ))
    val entitiesStruct = mockStructType(List(("entities", NAMED_ENTITY)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockNerData), entitiesStruct)

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

    val mockNerData = Seq(Row(
      List(Row(NAMED_ENTITY, 0, 5, "B-ORG", Map("entity" -> "United", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 7, 14, "O", Map("entity" -> "canceled", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 16, 18, "O", Map("entity" -> "the", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 20, 26, "B-TIME", Map("entity" -> "morning", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 28, 34, "O", Map("entity" -> "flights", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 36, 37, "O", Map("entity" -> "to", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 39, 45, "B-LOC", Map("entity" -> "Houston", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 47, 49, "O", Map("entity" -> "and", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 51, 56, "B-LOC", Map("entity" -> "Dallas", "sentence" -> "0"), List())
      )
    ))
    val entitiesStruct = mockStructType(List(("entities", NAMED_ENTITY)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockNerData), entitiesStruct)

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

    val mockNerData = Seq(Row(
      List(Row(NAMED_ENTITY, 0, 5, "B-ORG", Map("entity" -> "United", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 7, 14, "O", Map("entity" -> "canceled", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 16, 18, "O", Map("entity" -> "the", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 20, 26, "B-TIME", Map("entity" -> "morning", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 28, 34, "O", Map("entity" -> "flights", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 36, 37, "O", Map("entity" -> "to", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 39, 45, "B-LOC", Map("entity" -> "Houston", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 46, 46, "O", Map("entity" -> ".", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 48, 49, "O", Map("entity" -> "So", "sentence" -> "1"), List()),
        Row(NAMED_ENTITY, 50, 50, "O", Map("entity" -> ",", "sentence" -> "1"), List()),
        Row(NAMED_ENTITY, 52, 52, "O", Map("entity" -> "I", "sentence" -> "1"), List()),
        Row(NAMED_ENTITY, 54, 57, "O", Map("entity" -> "will", "sentence" -> "1"), List()),
        Row(NAMED_ENTITY, 59, 60, "O", Map("entity" -> "go", "sentence" -> "1"), List()),
        Row(NAMED_ENTITY, 62, 63, "O", Map("entity" -> "to", "sentence" -> "1"), List()),
        Row(NAMED_ENTITY, 65, 70, "B-LOC", Map("entity" -> "London", "sentence" -> "1"), List()),
        Row(NAMED_ENTITY, 72, 79, "B-TIME", Map("entity" -> "tomorrow", "sentence" -> "1"), List())
      )
    ))
    val entitiesStruct = mockStructType(List(("entities", NAMED_ENTITY)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockNerData), entitiesStruct)

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
