package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotator.{PerceptronApproach, Tokenizer}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.ml.Pipeline
import org.scalatest._

class FunctionsTestSpec extends FlatSpec {

  "functions in functions" should "work successfully" in {

    import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val pos = new PerceptronApproach()
      .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/", ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      .setNIterations(3)
      .setInputCols("document", "token")
      .setOutputCol("pos")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        pos
      ))

    val model = pipeline.fit(Seq.empty[String].toDF("text"))
    val data = model.transform(Seq("Peter is a very good and compromised person.").toDF("text"))

    import functions._

    val mapped = data.mapAnnotations("pos", "modpos", (annotations: Seq[Annotation]) => {
      annotations.filter(_.result == "JJ")
    })

    val modified = data.mapAnnotations("pos", "modpos", (_: Seq[Annotation]) => {
      "hello world"
    })

    val filtered = data.filterByAnnotations("pos", (annotations: Seq[Annotation]) => {
      annotations.exists(_.result == "JJ")
    })

    mapped.show(truncate = false)
    modified.show(truncate = false)
    filtered.show(truncate = false)
  }

}
