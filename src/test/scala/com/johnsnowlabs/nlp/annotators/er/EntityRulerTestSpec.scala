package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerTestSpec extends AnyFlatSpec with SparkSessionTest {

  "Entity Ruler" should "serialize" in {
    import spark.implicits._

    val textDataSet = Seq("John Snow is a good boss").toDS.toDF("text")
    val entityRuler = new EntityRuler()
      .setInputCols("document", "token")
      .setOutputCol("entities")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityRuler))

    val entityRulerModel = pipeline.fit(emptyDataSet)
    val resultDataSet = entityRulerModel.transform(textDataSet)
    resultDataSet.show()
    println("")
  }

}
