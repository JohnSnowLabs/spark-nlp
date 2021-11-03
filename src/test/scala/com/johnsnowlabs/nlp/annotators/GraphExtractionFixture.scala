/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
        Row(DEPENDENCY, 48, 49, "go", Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 50, 50, "go", Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 52, 52, "go", Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 54, 57, "go", Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 59, 60, "root", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 62, 63, "go", Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 65, 70, "go", Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
          "sentence" -> "1"), List()),
        Row(DEPENDENCY, 72, 79, "go", Map("head" -> "5", "head.begin" -> "59", "head.end" -> "60",
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

  def getOverlappingEntities(spark: SparkSession, pipeline: Pipeline): DataFrame = {
    import spark.implicits._
    val textDataSet = Seq("Peter Parker is a nice person and lives in New York")
      .toDS.toDF("text")
    val tokenDataSet = pipeline.fit(textDataSet).transform(textDataSet)
    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 4, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
        "sentence" -> "0"), List()),
        Row(DEPENDENCY, 6, 11, "person", Map("head" -> "6", "head.begin" -> "23", "head.end" -> "28",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 13, 14, "person", Map("head" -> "6", "head.begin" -> "23", "head.end" -> "28",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 16, 16, "person", Map("head" -> "6", "head.begin" -> "23", "head.end" -> "28",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 18, 21, "person", Map("head" -> "6", "head.begin" -> "23", "head.end" -> "28",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 23, 28, "Peter", Map("head" -> "1", "head.begin" -> "0", "head.end" -> "4",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 30, 32, "lives", Map("head" -> "8", "head.begin" -> "34", "head.end" -> "38",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 34, 38, "person", Map("head" -> "6", "head.begin" -> "23", "head.end" -> "28",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 40, 41, "York", Map("head" -> "11", "head.begin" -> "47", "head.end" -> "50",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 43, 45, "York", Map("head" -> "11", "head.begin" -> "47", "head.end" -> "50",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 47, 50, "person", Map("head" -> "6", "head.begin" -> "23", "head.end" -> "28",
          "sentence" -> "0"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 4, "root", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 6, 11, "flat", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 13, 14, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 16, 16, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 18, 21, "amod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 23, 28, "flat", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 30, 32, "cc", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 34, 38, "flat", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 40, 41, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 43, 45, "flat", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 47, 50, "flat", Map("sentence" -> "0"), List())
      )
    ))

    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))
    val mockDependencyParserDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockDependencyParserData),
      dependenciesStruct)

    val mockNerData = Seq(Row(
      List(Row(NAMED_ENTITY, 0, 4, "B-PER", Map("entity" -> "Peter", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 6, 11, "I-PER", Map("entity" -> "Parker", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 13, 14, "O", Map("entity" -> "is", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 16, 16, "O", Map("entity" -> "a", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 18, 21, "O", Map("entity" -> "nice", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 23, 28, "O", Map("entity" -> "person", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 30, 32, "O", Map("entity" -> "and", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 34, 38, "O", Map("entity" -> "lives", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 40, 41, "O", Map("entity" -> "in", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 43, 45, "B-LOC", Map("entity" -> "New", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 47, 50, "I-LOC", Map("entity" -> "York", "sentence" -> "0"), List())
      )
    ))

    val entitiesStruct = mockStructType(List(("entities", NAMED_ENTITY)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockNerData), entitiesStruct)

    val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

    tokenDataSet.join(mockAnnotatorsDataSet)

  }

  def getDeepEntities(spark: SparkSession, pipeline: Pipeline): DataFrame = {
    import spark.implicits._
    val textDataSet = Seq("Later tonight, by the time John sees Bill and Mary goes to Pasadena")
      .toDS.toDF("text")
    val tokenDataSet = pipeline.fit(textDataSet).transform(textDataSet)
    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 4, "tonight", Map("head" -> "2", "head.begin" -> "6", "head.end" -> "12",
        "sentence" -> "0"), List()),
        Row(DEPENDENCY, 6, 12, "sees", Map("head" -> "8", "head.begin" -> "32", "head.end" -> "35",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 13, 13, "sees", Map("head" -> "8", "head.begin" -> "32", "head.end" -> "35",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 15, 16, "time", Map("head" -> "6", "head.begin" -> "22", "head.end" -> "25",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 18, 20, "time", Map("head" -> "6", "head.begin" -> "22", "head.end" -> "25",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 22, 25, "sees", Map("head" -> "8", "head.begin" -> "32", "head.end" -> "35",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 27, 30, "sees", Map("head" -> "8", "head.begin" -> "32", "head.end" -> "35",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 32, 35, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 37, 40, "goes", Map("head" -> "12", "head.begin" -> "51", "head.end" -> "54",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 42, 44, "Mary", Map("head" -> "11", "head.begin" -> "46", "head.end" -> "49",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 46, 49, "Bill", Map("head" -> "9", "head.begin" -> "37", "head.end" -> "40",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 51, 54, "sees", Map("head" -> "8", "head.begin" -> "32", "head.end" -> "35",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 56, 57, "Pasadena", Map("head" -> "14", "head.begin" -> "59", "head.end" -> "66",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 59, 66, "goes", Map("head" -> "12", "head.begin" -> "51", "head.end" -> "54",
          "sentence" -> "0"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 4, "advmod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 6, 12, "obl:tmod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 13, 13, "punct", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 15, 16, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 18, 20, "det", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 22, 25, "obl", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 27, 30, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 32, 35, "root", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 37, 40, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 42, 44, "cc", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 46, 49, "conj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 51, 54, "ccomp", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 56, 57, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 59, 66, "obl", Map("sentence" -> "0"), List())
      )
    ))

    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))
    val mockDependencyParserDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockDependencyParserData),
      dependenciesStruct)

    val mockNerData = Seq(Row(
      List(Row(NAMED_ENTITY, 0, 4, "O", Map("entity" -> "Later", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 6, 12, "O", Map("entity" -> "tonight", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 13, 13, "O", Map("entity" -> ",", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 15, 16, "O", Map("entity" -> "by", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 18, 20, "O", Map("entity" -> "the", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 22, 25, "O", Map("entity" -> "time", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 27, 30, "B-PER", Map("entity" -> "John", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 32, 35, "O", Map("entity" -> "sees", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 37, 40, "B-PER", Map("entity" -> "Bill", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 42, 44, "O", Map("entity" -> "and", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 46, 49, "B-PER", Map("entity" -> "Mary", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 51, 54, "O", Map("entity" -> "goes", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 56, 57, "O", Map("entity" -> "to", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 59, 66, "B-LOC", Map("entity" -> "Pasadena", "sentence" -> "0"), List())
      )
    ))

    val entitiesStruct = mockStructType(List(("entities", NAMED_ENTITY)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockNerData), entitiesStruct)

    val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

    tokenDataSet.join(mockAnnotatorsDataSet)

  }

  def getPubTatorEntities(spark: SparkSession, pipeline: Pipeline): DataFrame = {
    import spark.implicits._
    val textDataSet = Seq("Influence of interleukin-6 gene polymorphisms on coronary_artery_calcification in patients with psoriasis")
      .toDS.toDF("text")
    val tokenDataSet = pipeline.fit(textDataSet).transform(textDataSet)
    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 8, "polymorphisms", Map("head" -> "5", "head.begin" -> "32", "head.end" -> "44",
        "sentence" -> "0"), List()),
        Row(DEPENDENCY, 10, 11, "gene", Map("head" -> "4", "head.begin" -> "27", "head.end" -> "30",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 13, 25, "gene", Map("head" -> "4", "head.begin" -> "27", "head.end" -> "30",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 27, 30, "Influence", Map("head" -> "1", "head.begin" -> "0", "head.end" -> "8",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 32, 44, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 46, 47, "coronary_artery_calcification", Map("head" -> "7", "head.begin" -> "49", "head.end" -> "77",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 49, 77, "polymorphisms", Map("head" -> "5", "head.begin" -> "32", "head.end" -> "44",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 79, 80, "patients", Map("head" -> "9", "head.begin" -> "82", "head.end" -> "89",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 82, 89, "coronary artery calcification", Map("head" -> "7", "head.begin" -> "49", "head.end" -> "77",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 91, 94, "psoriasis", Map("head" -> "11", "head.begin" -> "96", "head.end" -> "104",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 96, 104, "patients", Map("head" -> "9", "head.begin" -> "82", "head.end" -> "89",
          "sentence" -> "0"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 8, "nsubj", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 10, 11, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 13, 25, "amod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 27, 30, "nmod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 32, 44, "root", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 46, 47, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 47, 77, "nmod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 79, 80, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 82, 89, "nmod", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 91, 94, "case", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 96, 104, "nmod", Map("sentence" -> "0"), List())
      )
    ))

    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))
    val mockDependencyParserDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockDependencyParserData),
      dependenciesStruct)

    val mockNerData = Seq(Row(
      List(Row(NAMED_ENTITY, 0, 8, "O", Map("entity" -> "Influence", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 10, 11, "O", Map("entity" -> "of", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 13, 25, "B-GENE", Map("entity" -> "interleukin-6", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 27, 30, "O", Map("entity" -> "gene", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 32, 44, "O", Map("entity" -> "polymorphisms", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 46, 47, "O", Map("entity" -> "on", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 47, 77, "B-DISEASE", Map("entity" -> "coronary_artery_calcification", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 79, 80, "O", Map("entity" -> "in", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 82, 89, "O", Map("entity" -> "patients", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 91, 94, "O", Map("entity" -> "with", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 96, 104, "O", Map("entity" -> "psoriasis", "sentence" -> "0"), List())
      )
    ))

    val entitiesStruct = mockStructType(List(("entities", NAMED_ENTITY)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockNerData), entitiesStruct)

    val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

    tokenDataSet.join(mockAnnotatorsDataSet)

  }

  def getEntitiesWithNoTypeParserOutput(spark: SparkSession, pipeline: Pipeline): DataFrame = {
    import spark.implicits._
    val textDataSet = Seq("Ms. Stewart is taking two pills of paracetamol a day due to her heart disease")
      .toDS.toDF("text")
    val tokenDataSet = pipeline.fit(textDataSet).transform(textDataSet)
    val mockDependencyParserData = Seq(Row(
      List(Row(DEPENDENCY, 0, 1, "taking", Map("head" -> "5", "head.begin" -> "15", "head.end" -> "20",
        "sentence" -> "0"), List()),
        Row(DEPENDENCY, 2, 2, "Ms", Map("head" -> "1", "head.begin" -> "0", "head.end" -> "1",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 4, 10, "taking", Map("head" -> "5", "head.begin" -> "15", "head.end" -> "20",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 12, 13, "taking", Map("head" -> "5", "head.begin" -> "15", "head.end" -> "20",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 15, 20, "ROOT", Map("head" -> "0", "head.begin" -> "-1", "head.end" -> "-1",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 22, 24, "pills", Map("head" -> "7", "head.begin" -> "26", "head.end" -> "30",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 26, 30, "taking", Map("head" -> "5", "head.begin" -> "15", "head.end" -> "20",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 32, 33, "paracetamol", Map("head" -> "9", "head.begin" -> "35", "head.end" -> "45",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 35, 45, "pills", Map("head" -> "7", "head.begin" -> "26", "head.end" -> "30",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 47, 47, "day", Map("head" -> "11", "head.begin" -> "49", "head.end" -> "51",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 49, 51, "paracetamol", Map("head" -> "9", "head.begin" -> "35", "head.end" -> "45",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 53, 55, "disease", Map("head" -> "16", "head.begin" -> "70", "head.end" -> "76",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 57, 58, "due", Map("head" -> "12", "head.begin" -> "53", "head.end" -> "55",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 60, 62, "to", Map("head" -> "13", "head.begin" -> "57", "head.end" -> "58",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 64, 68, "to", Map("head" -> "13", "head.begin" -> "57", "head.end" -> "58",
          "sentence" -> "0"), List()),
        Row(DEPENDENCY, 70, 76, "taking", Map("head" -> "5", "head.begin" -> "15", "head.end" -> "20",
          "sentence" -> "0"), List())
      ),
      List(Row(LABELED_DEPENDENCY, 0, 1, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 2, 2, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 4, 10, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 12, 13, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 15, 20, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 22, 24, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 26, 30, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 32, 33, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 35, 45, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 47, 47, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 49, 51, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 53, 55, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 57, 58, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 60, 62, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 64, 68, "<no-type>", Map("sentence" -> "0"), List()),
        Row(LABELED_DEPENDENCY, 70, 76, "<no-type>", Map("sentence" -> "0"), List())
      )
    ))

    val dependenciesStruct = mockStructType(List(("heads", DEPENDENCY), ("deprel", LABELED_DEPENDENCY)))
    val mockDependencyParserDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockDependencyParserData),
      dependenciesStruct)

    val mockNerData = Seq(Row(
      List(Row(NAMED_ENTITY, 0, 1, "O", Map("entity" -> "Peter", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 2, 2, "O", Map("entity" -> "Parker", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 4, 10, "O", Map("entity" -> "is", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 12, 13, "O", Map("entity" -> "a", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 15, 20, "O", Map("entity" -> "nice", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 22, 24, "O", Map("entity" -> "person", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 26, 30, "O", Map("entity" -> "and", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 32, 33, "O", Map("entity" -> "lives", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 35, 45, "I-Medication", Map("entity" -> "in", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 47, 47, "O", Map("entity" -> "New", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 49, 51, "O", Map("entity" -> "York", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 53, 55, "O", Map("entity" -> "York", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 57, 58, "O", Map("entity" -> "York", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 60, 62, "O", Map("entity" -> "York", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 64, 68, "I-Diagnosis", Map("entity" -> "York", "sentence" -> "0"), List()),
        Row(NAMED_ENTITY, 70, 76, "I-Diagnosis", Map("entity" -> "York", "sentence" -> "0"), List())
      )
    ))

    val entitiesStruct = mockStructType(List(("entities", NAMED_ENTITY)))
    val mockEntitiesDataSet = spark.createDataFrame(spark.sparkContext.parallelize(mockNerData), entitiesStruct)

    val mockAnnotatorsDataSet = mockDependencyParserDataSet.join(mockEntitiesDataSet)

    tokenDataSet.join(mockAnnotatorsDataSet)
  }

  private def mockStructType(columnsAndAnnotators: List[(String, String)]): StructType = {
    val structFields: List[StructField] = columnsAndAnnotators.map { columnAndAnnotator =>
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
