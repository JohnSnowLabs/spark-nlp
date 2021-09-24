package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  private val textDataSet = Seq("Aberforth Dumbledore is a good wizard").toDS.toDF("text")


  "Entity Ruler" should "raise an error when patterns resource is not defined" in {
    val entityRuler = new EntityRuler()
      .setInputCols("document", "token")
      .setOutputCol("entities")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    assertThrows[IllegalArgumentException] {
      pipeline.fit(emptyDataSet)
    }
  }

  it should "train" in {

    val entityRuler = new EntityRuler()
      .setInputCols("document", "token")
      .setOutputCol("entities")
      .setPatterns("src/test/resources/entity-ruler/patterns.json", ReadAs.TEXT)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    val entityRulerModel = pipeline.fit(emptyDataSet)
    val resultDataSet = entityRulerModel.transform(textDataSet)
    resultDataSet.show()
    println("")
  }

}
