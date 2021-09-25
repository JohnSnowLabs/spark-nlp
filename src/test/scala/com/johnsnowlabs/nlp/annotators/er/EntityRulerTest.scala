package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.{Annotation, AssertAnnotations}
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, NODE}
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "Entity Ruler" should "raise an error when patterns resource is not defined" in {
    val entityRuler = new EntityRuler()
      .setInputCols("document", "token")
      .setOutputCol("entities")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(emptyDataSet)
    }
  }

  it should "train an entity ruler model" in {
    val textDataSet = Seq("John Snow is a good boss").toDS.toDF("text")
    val entityRuler = new EntityRuler()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatterns("src/test/resources/entity-ruler/patterns.json", ReadAs.TEXT)

    val entityRulerModel = entityRuler.fit(textDataSet)

    assert(entityRulerModel != null)
    assert(entityRulerModel.isInstanceOf[EntityRulerModel])
  }

  "An Entity Ruler model" should "infer entities" in {
    val textDataSet = Seq("John Snow lives in Winterfell").toDS.toDF("text")
    val entityRuler = new EntityRuler()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatterns("src/test/resources/entity-ruler/patterns.json", ReadAs.TEXT)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))
    val entityRulerModel = pipeline.fit(emptyDataSet)
    val expectedEntities = Array(Seq(
      Annotation(CHUNK, 0, 3, "John", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 5, 8, "Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "sentence" -> "0"))
    ))

    val resultDataSet = entityRulerModel.transform(textDataSet)
    val actualEntities = AssertAnnotations.getActualResult(resultDataSet, "entities")

    AssertAnnotations.assertFields(expectedEntities, actualEntities)
  }

}
