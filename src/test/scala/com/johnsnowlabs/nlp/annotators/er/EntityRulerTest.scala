package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
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
  }

  it should "raise an error when file is csv and delimiter is not set" taggedAs FastTest in {
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.csv", ReadAs.TEXT, Map("format" -> "csv"))

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
      .setPatternsResource("src/test/resources/entity-ruler/patterns.csv", ReadAs.TEXT,
        Map("format" -> "myFormat"))

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(emptyDataSet)
    }
  }

  it should "train an entity ruler model" taggedAs FastTest in {
    val textDataSet = Seq("John Snow is a good boss").toDS.toDF("text")
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.json", ReadAs.TEXT)

    val entityRulerModel = entityRuler.fit(textDataSet)

    assert(entityRulerModel != null)
    assert(entityRulerModel.isInstanceOf[EntityRulerModel])
  }

  "An Entity Ruler model" should "infer entities when reading JSON file" taggedAs FastTest in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    tokenizer.setExceptions(Array("John Snow"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.json", ReadAs.TEXT)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 0, 8, "John Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading CSV file" taggedAs FastTest in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    tokenizer.setExceptions(Array("John Snow"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.csv", ReadAs.TEXT,
        Map("format" -> "csv", "delimiter" -> "\\|"))
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 0, 8, "John Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading CSV file with Spark" taggedAs FastTest in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    tokenizer.setExceptions(Array("John Snow"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.csv", ReadAs.SPARK,
        Map("format" -> "csv", "delimiter" -> "|"))
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 0, 8, "John Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark" taggedAs FastTest in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    tokenizer.setExceptions(Array("John Snow"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.json", ReadAs.SPARK, Map("format" -> "json"))
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 0, 8, "John Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file" taggedAs FastTest in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    tokenizer.setExceptions(Array("John Snow"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.jsonl", ReadAs.TEXT, Map("format" -> "jsonl"))
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 0, 8, "John Snow", Map("entity" -> "PERSON", "id" -> "names-with-j", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "id" -> "locations", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file with Spark" taggedAs FastTest in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    tokenizer.setExceptions(Array("John Snow"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.jsonl", ReadAs.SPARK, Map("format" -> "jsonl"))
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 0, 8, "John Snow", Map("entity" -> "PERSON", "id" -> "names-with-j", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "id" -> "locations", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  "An Entity Ruler model with regex patterns" should "infer entities when reading JSON file" taggedAs FastTest in {
    val textDataSet = Seq("Lord Eddard Stark was the head of House Stark").toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/regex_patterns.json", ReadAs.TEXT)
      .setEnablePatternRegex(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 5, 16, "Eddard Stark", Map("entity" -> "PERSON", "id" -> "person-regex", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading CSV file" taggedAs FastTest in {
    val textDataSet = Seq("Lord Eddard Stark was the head of House Stark").toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/regex_patterns.csv", ReadAs.TEXT,
        Map("format" -> "csv", "delimiter" -> "\\|"))
      .setEnablePatternRegex(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 5, 16, "Eddard Stark", Map("entity" -> "PERSON", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading CSV file with Spark" taggedAs FastTest in {
    val textDataSet = Seq("Lord Eddard Stark was the head of House Stark").toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/regex_patterns.csv", ReadAs.SPARK,
        Map("format" -> "csv", "delimiter" -> "|"))
      .setEnablePatternRegex(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 5, 16, "Eddard Stark", Map("entity" -> "PERSON", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading JSON file with Spark" taggedAs FastTest in {
    val textDataSet = Seq("Lord Eddard Stark was the head of House Stark").toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/regex_patterns.json", ReadAs.SPARK, Map("format" -> "json"))
      .setEnablePatternRegex(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 5, 16, "Eddard Stark", Map("entity" -> "PERSON", "id" -> "person-regex", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) file" taggedAs FastTest in {
    val textDataSet = Seq("Lord Eddard Stark was the head of House Stark").toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/regex_patterns.jsonl", ReadAs.TEXT, Map("format" -> "jsonl"))
      .setEnablePatternRegex(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 5, 16, "Eddard Stark", Map("entity" -> "PERSON", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  it should "infer entities when reading JSON Lines (JSONL) with Spark" taggedAs FastTest in {
    val textDataSet = Seq("Lord Eddard Stark was the head of House Stark").toDS.toDF("text")
    tokenizer.setExceptions(Array("Eddard Stark"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/regex_patterns.jsonl", ReadAs.SPARK, Map("format" -> "jsonl"))
      .setEnablePatternRegex(true)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 5, 16, "Eddard Stark", Map("entity" -> "PERSON", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

  "EntityRuler with memory not optimized" should "raise an error when file is csv and delimiter is not set" taggedAs FastTest in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    tokenizer.setExceptions(Array("John Snow"))
    val entityRuler = new EntityRulerApproach()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatternsResource("src/test/resources/entity-ruler/patterns.csv", ReadAs.TEXT,
        Map("format" -> "csv", "delimiter" -> "\\|"))
      .setOptimizeMemory(false)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    val result = pipeline.fit(emptyDataSet).transform(textDataSet)

    result.show(false)
  }

}
