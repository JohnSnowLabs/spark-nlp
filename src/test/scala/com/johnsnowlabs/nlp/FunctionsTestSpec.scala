package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotator.{PerceptronApproach, Tokenizer}
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.ArrayType
import org.scalatest._

class FunctionsTestSpec extends FlatSpec {

  "functions in functions" should "work successfully" taggedAs FastTest in {

    import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._

    val trainingPerceptronDF = POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/", "|", "tags")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val pos = new PerceptronApproach()
      .setInputCols("document", "token")
      .setOutputCol("pos")
      .setPosColumn("tags")
      .setNIterations(3)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        pos
      ))

    val model = pipeline.fit(trainingPerceptronDF)
    val data = model.transform(Seq("Peter is a very good and compromised person.").toDF("text"))

    import functions._

    val mapped = data.mapAnnotationsCol("pos", "modpos", (annotations: Seq[Annotation]) => {
      annotations.filter(_.result == "JJ")
    })

    val modified = data.mapAnnotationsCol("pos", "modpos", (_: Seq[Annotation]) => {
      "hello world"
    })

    val filtered = data.filterByAnnotationsCol("pos", (annotations: Seq[Annotation]) => {
      annotations.exists(_.result == "JJ")
    })

    import org.apache.spark.sql.functions.col

    val udfed = data.select(mapAnnotations((annotations: Seq[Annotation]) => {
      annotations.filter(_.result == "JJ")
    })(col("pos")))

    val udfed2 = data.select(mapAnnotationsStrict((annotations: Seq[Annotation]) => {
      annotations.filter(_.result == "JJ")
    })(col("pos")))

    mapped.show(1)
    modified.show(1)
    filtered.show(1)
    udfed.show(1)
    udfed2.show(1)
  }

}
