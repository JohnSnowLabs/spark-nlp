/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.AssertAnnotations
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.er.EntityRulerFixture._
import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "Entity Ruler" should "raise an error when patterns resource is not set" taggedAs FastTest in {
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(emptyDataSet)
    }

    val entityRulerKeywords = new EntityRulerApproach()
      .setInputCols("document")
      .setOutputCol("entities")

    val pipelineKeywords = new Pipeline().setStages(Array(documentAssembler, entityRulerKeywords))

    assertThrows[IllegalArgumentException] {
      pipelineKeywords.fit(emptyDataSet)
    }
  }

  it should "raise an error when file is csv and delimiter is not set" taggedAs FastTest in {
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource(
        "src/test/resources/entity-ruler/keywords_with_regex_field.csv",
        ReadAs.TEXT,
        Map("format" -> "csv"))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(emptyDataSet)
    }
  }

  it should "raise an error when path file is not set" taggedAs FastTest in {
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("", ReadAs.SPARK, Map("format" -> "csv"))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(emptyDataSet)
    }
  }

  it should "raise an error when unknown file formats are set" taggedAs FastTest in {
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource(
        "src/test/resources/entity-ruler/keywords_with_regex_field.csv",
        ReadAs.TEXT,
        Map("format" -> "myFormat"))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(emptyDataSet)
    }
  }

  it should "train an entity ruler model" taggedAs FastTest in {
    val textDataSet = Seq("John Snow is a good boss").toDS.toDF("text")
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/keywords_only.json", ReadAs.TEXT)

    val entityRulerModel = entityRuler.fit(textDataSet)

    assert(entityRulerModel != null)
    assert(entityRulerModel.isInstanceOf[EntityRulerModel])
  }

  private val testPath = "src/test/resources/entity-ruler"

  "An Entity Ruler model" should "infer entities when reading JSON file" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.TEXT, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading CSV file" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.TEXT,
      Map("format" -> "csv", "delimiter" -> "\\|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading CSV file without regex with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_without_regex_field.csv",
      ReadAs.SPARK,
      Map("format" -> "csv", "delimiter" -> ","))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading CSV file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.SPARK,
      Map("format" -> "csv", "delimiter" -> "|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.SPARK, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_with_id.jsonl", ReadAs.TEXT, Map("format" -> "jsonl"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText1, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_with_id.jsonl",
        ReadAs.SPARK,
        Map("format" -> "jsonl"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText1, actualEntities)
  }

  "An Entity Ruler model with regex patterns" should "infer entities when reading JSON file" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizerWithSentence.setExceptions(Array("Eddard Stark"))
    val externalResource =
      ExternalResource(s"$testPath/keywords_regex.json", ReadAs.TEXT, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText2, actualEntities)
  }

  it should "infer entities when reading CSV file" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizerWithSentence.setExceptions(Array("Eddard Stark"))
    val externalResource = ExternalResource(
      s"$testPath/keywords_regex_with_regex_field.csv",
      ReadAs.TEXT,
      Map("format" -> "csv", "delimiter" -> "\\|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  it should "infer entities when reading CSV file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizerWithSentence.setExceptions(Array("Eddard Stark"))
    val externalResource = ExternalResource(
      s"$testPath/keywords_regex_with_regex_field.csv",
      ReadAs.SPARK,
      Map("format" -> "csv", "delimiter" -> "|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizerWithSentence.setExceptions(Array("Eddard Stark"))
    val externalResource =
      ExternalResource(s"$testPath/keywords_regex.json", ReadAs.SPARK, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText2, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizerWithSentence.setExceptions(Array("Eddard Stark"))
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_regex_without_id.jsonl",
        ReadAs.TEXT,
        Map("format" -> "jsonl"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizerWithSentence.setExceptions(Array("Eddard Stark"))
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_regex_without_id.jsonl",
        ReadAs.SPARK,
        Map("format" -> "jsonl"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  "An Entity Ruler model without using storage" should "infer entities when reading JSON file" in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.TEXT, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading CSV file" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.TEXT,
      Map("format" -> "csv", "delimiter" -> "\\|"))
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading CSV file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.SPARK,
      Map("format" -> "csv", "delimiter" -> "|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.SPARK, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_with_id.jsonl", ReadAs.TEXT, Map("format" -> "jsonl"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText1, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_with_id.jsonl",
        ReadAs.SPARK,
        Map("format" -> "jsonl"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText1, actualEntities)
  }

  "An Entity Ruler model without using storage with regex patterns" should "infer entities when reading JSON file" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_regex.json", ReadAs.TEXT, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText2, actualEntities)
  }

  it should "infer entities when reading CSV file" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_regex_with_regex_field.csv",
      ReadAs.TEXT,
      Map("format" -> "csv", "delimiter" -> "\\|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  it should "infer entities when reading CSV file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_regex_with_regex_field.csv",
      ReadAs.SPARK,
      Map("format" -> "csv", "delimiter" -> "|"))
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_regex.json", ReadAs.SPARK, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText2, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_regex_without_id.jsonl",
        ReadAs.TEXT,
        Map("format" -> "jsonl"))
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) with Spark" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_regex_without_id.jsonl",
        ReadAs.SPARK,
        Map("format" -> "jsonl"))
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText2, actualEntities)
  }

  "An Entity Ruler" should "serialize and deserialize a model" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource(s"$testPath/keywords_regex.json", readAs = ReadAs.TEXT)
    val entityRulerModel = entityRuler.fit(emptyDataSet)

    entityRulerModel.write.overwrite().save("tmp_entity_ruler_model_storage")
    val loadedEntityRulerModel = EntityRulerModel.load("tmp_entity_ruler_model_storage")
    val entityRulerPipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, loadedEntityRulerModel))

    val resultDataSet = entityRulerPipeline.fit(emptyDataSet).transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText2, actualEntities)
  }

  it should "serialize and deserialize a model without storage" taggedAs FastTest in {
    val textDataSet = Seq(text2).toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource(s"$testPath/keywords_regex.json", readAs = ReadAs.TEXT)
      .setUseStorage(false)
    val entityRulerModel = entityRuler.fit(emptyDataSet)

    entityRulerModel.write.overwrite().save("tmp_entity_ruler_model")
    val loadedEntityRulerModel = EntityRulerModel.load("tmp_entity_ruler_model")
    val entityRulerPipeline =
      new Pipeline().setStages(Array(documentAssembler, tokenizer, loadedEntityRulerModel))

    val resultDataSet = entityRulerPipeline.fit(emptyDataSet).transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText2, actualEntities)
  }

  it should "serialize and deserialize a model without regex" taggedAs FastTest in {
    val textDataSet = Seq(text1).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.TEXT, Map("format" -> "json"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document")
      .setOutputCol("entities")
      .setPatternsResource(
        externalResource.path,
        externalResource.readAs,
        externalResource.options)
    val entityRulerModel = entityRuler.fit(emptyDataSet)

    entityRulerModel.write.overwrite().save("tmp_entity_ruler_model_storage")
    val loadedEntityRulerModel = EntityRulerModel.load("tmp_entity_ruler_model_storage")
    val entityRulerPipeline =
      new Pipeline().setStages(Array(documentAssembler, loadedEntityRulerModel))

    val resultDataSet = entityRulerPipeline.fit(emptyDataSet).transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText1, actualEntities)
  }

  "An Entity Ruler model at sentence level" should "infer entities when reading JSON file" taggedAs FastTest in {
    val textDataSet = Seq(text3).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.TEXT, Map("format" -> "json"))

    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText3, actualEntities)
  }

  it should "infer entities when reading JSON file as Spark" taggedAs FastTest in {
    val textDataSet = Seq(text3).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.SPARK, Map("format" -> "json"))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText3, actualEntities)
  }

  it should "infer entities when reading JSON file without storage" taggedAs FastTest in {
    val textDataSet = Seq(text3).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.TEXT, Map("format" -> "json"))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText3, actualEntities)
  }

  it should "infer entities when reading JSON file without storage and Spark" taggedAs FastTest in {
    val textDataSet = Seq(text3).toDS
      .toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.SPARK, Map("format" -> "json"))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText3, actualEntities)
  }

  it should "infer entities when reading JSON lines file" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS
      .toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_with_id.jsonl", ReadAs.TEXT, Map("format" -> "jsonl"))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText4, actualEntities)
  }

  it should "infer entities when reading JSON lines file as Spark" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS.toDF("text")
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_with_id.jsonl",
        ReadAs.SPARK,
        Map("format" -> "jsonl"))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText4, actualEntities)
  }

  it should "infer entities when reading JSON lines file without storage" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/keywords_with_id.jsonl", ReadAs.TEXT, Map("format" -> "jsonl"))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText4, actualEntities)
  }

  it should "infer entities when reading JSON lines file without storage and Spark" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS.toDF("text")
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_with_id.jsonl",
        ReadAs.SPARK,
        Map("format" -> "jsonl"))
    val entityRulerPipeline =
      getEntityRulerKeywordsPipeline(externalResource, sentenceMatch = true, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesWithIdFromText4, actualEntities)
  }

  it should "infer entities when reading CSV file" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.TEXT,
      Map("format" -> "csv", "delimiter" -> "\\|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, sentenceMatch = true)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText4, actualEntities)
  }

  it should "infer entities when reading CSV file as Spark" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.SPARK,
      Map("format" -> "csv", "delimiter" -> "|"))
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, sentenceMatch = true)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText4, actualEntities)
  }

  it should "infer entities when reading CSV file without storage" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.TEXT,
      Map("format" -> "csv", "delimiter" -> "\\|"))
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, sentenceMatch = true, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText4, actualEntities)
  }

  it should "infer entities when reading CSV file without storage and Spark" taggedAs FastTest in {
    val textDataSet = Seq(text4).toDS.toDF("text")
    val externalResource = ExternalResource(
      s"$testPath/keywords_with_regex_field.csv",
      ReadAs.SPARK,
      Map("format" -> "csv", "delimiter" -> "|"))
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, sentenceMatch = true, useStorage = false)

    val resultDataSet = entityRulerPipeline.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText4, actualEntities)
  }

  it should "infer entities when reading JSON file with only regex and matching at sentence level" taggedAs FastTest in {
    val textDataSet = Seq(text5).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/regex_only.json", ReadAs.TEXT, Map("format" -> "JSON"))
    val entityRulerPipelineWithUseStorage =
      getEntityRulerRegexPipeline(externalResource, sentenceMatch = true)
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, sentenceMatch = true, useStorage = false)

    var resultDataSet = entityRulerPipelineWithUseStorage.transform(textDataSet)
    var actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesSentenceLevelFromText5, actualEntities)

    resultDataSet = entityRulerPipeline.transform(textDataSet)
    actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesSentenceLevelFromText5, actualEntities)
  }

  it should "infer entities when reading JSON file with only regex and matching at token level" taggedAs FastTest in {
    val textDataSet = Seq(text5).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/regex_only.json", ReadAs.TEXT, Map("format" -> "JSON"))
    val entityRulerPipelineWithUseStorage = getEntityRulerRegexPipeline(externalResource)
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, useStorage = false)

    var resultDataSet = entityRulerPipelineWithUseStorage.transform(textDataSet)
    var actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText5, actualEntities)

    resultDataSet = entityRulerPipeline.transform(textDataSet)
    actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText5, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark only regex and matching at sentence level" taggedAs FastTest in {
    val textDataSet = Seq(text5).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/regex_only.json", ReadAs.SPARK, Map("format" -> "JSON"))
    val entityRulerPipelineWithUseStorage =
      getEntityRulerRegexPipeline(externalResource, sentenceMatch = true)
    val entityRulerPipeline =
      getEntityRulerRegexPipeline(externalResource, sentenceMatch = true, useStorage = false)

    var resultDataSet = entityRulerPipelineWithUseStorage.transform(textDataSet)
    var actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesSentenceLevelFromText5, actualEntities)

    resultDataSet = entityRulerPipeline.transform(textDataSet)
    actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesSentenceLevelFromText5, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark only regex and matching at token level" taggedAs FastTest in {
    val textDataSet = Seq(text5).toDS.toDF("text")
    val externalResource =
      ExternalResource(s"$testPath/regex_only.json", ReadAs.SPARK, Map("format" -> "JSON"))
    val entityRulerPipelineWithUseStorage = getEntityRulerRegexPipeline(externalResource)
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, useStorage = false)

    var resultDataSet = entityRulerPipelineWithUseStorage.transform(textDataSet)
    var actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText5, actualEntities)

    resultDataSet = entityRulerPipeline.transform(textDataSet)
    actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText5, actualEntities)
  }

  it should "infer entities for keywords and regex alike" taggedAs FastTest in {
    val textDataSet = Seq(text6).toDS.toDF("text")
    val externalResource =
      ExternalResource(
        s"$testPath/keywords_regex_with_id.json",
        ReadAs.SPARK,
        Map("format" -> "JSON"))
    val entityRulerPipelineWithUseStorage = getEntityRulerRegexPipeline(externalResource)
    val entityRulerPipeline = getEntityRulerRegexPipeline(externalResource, useStorage = false)

    var resultDataSet = entityRulerPipelineWithUseStorage.transform(textDataSet)
    var actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText6, actualEntities)

    resultDataSet = entityRulerPipeline.transform(textDataSet)
    actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntitiesFromText6, actualEntities)
  }

  it should "work with LightPipeline" taggedAs FastTest in {
    val externalResource =
      ExternalResource(s"$testPath/keywords_only.json", ReadAs.TEXT, Map("format" -> "json"))
    val entityRulerPipeline = getEntityRulerKeywordsPipeline(externalResource, useStorage = false)
    val lightPipeline = new LightPipeline(entityRulerPipeline)

    val actualResult = lightPipeline.annotate(text1)

    val expectedResult = Map(
      "document" -> Seq(text1),
      "sentence" -> Seq(text1),
      "entities" -> Seq("John Snow", "Winterfell"))
    assert(expectedResult == actualResult)
  }

  private def getEntityRulerRegexPipeline(
      externalResource: ExternalResource,
      sentenceMatch: Boolean = false,
      useStorage: Boolean = true): PipelineModel = {

    val entityRuler = new EntityRulerApproach()
      .setInputCols("sentence", "token")
      .setOutputCol("entities")
      .setPatternsResource(
        externalResource.path,
        externalResource.readAs,
        externalResource.options)
      .setUseStorage(useStorage)
      .setSentenceMatch(sentenceMatch)

    val pipeline = new Pipeline().setStages(
      Array(documentAssembler, sentenceDetector, tokenizerWithSentence, entityRuler))
    val entityRulerPipeline = pipeline.fit(emptyDataSet)

    entityRulerPipeline
  }

  private def getEntityRulerKeywordsPipeline(
      externalResource: ExternalResource,
      sentenceMatch: Boolean = false,
      useStorage: Boolean = true): PipelineModel = {

    val entityRuler = new EntityRulerApproach()
      .setInputCols("sentence")
      .setOutputCol("entities")
      .setPatternsResource(
        externalResource.path,
        externalResource.readAs,
        externalResource.options)
      .setUseStorage(useStorage)
      .setSentenceMatch(sentenceMatch)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, sentenceDetector, entityRuler))
    val entityRulerPipeline = pipeline.fit(emptyDataSet)

    entityRulerPipeline
  }

}
