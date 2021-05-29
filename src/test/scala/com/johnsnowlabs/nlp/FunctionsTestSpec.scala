package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotator.{PerceptronApproach, Tokenizer}
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.ArrayType
import org.junit.Assert.assertEquals
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

    val mapped = data.mapAnnotationsCol("pos", "modpos","pos", (annotations: Seq[Annotation]) => {
      annotations.filter(_.result == "JJ")
    })

    val modified = data.mapAnnotationsCol("pos", "modpos","pos", (_: Seq[Annotation]) => {
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


  "A mapAnnotationsCol" should "transform document to lower case" in {
    import SparkAccessor.spark.implicits._
    val df  = Seq(
      ("Pablito clavo un palito"),
      ("Un clavito chiquitillo")
    ).toDS.toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")


    val lower = (annotations: Seq[Annotation]) => {
      annotations.map(an => an.copy(result = an.result.toLowerCase))
    }
    import functions._

    val lowerDf = documentAssembler.transform(df.select("text")).mapAnnotationsCol[Seq[Annotation]](Seq("document"),"tail_document","document",lower)
    val tail_annotation = Annotation.collect(lowerDf, "tail_document").flatten.toSeq.sortBy(_.begin)
    assertEquals(tail_annotation.head.result, "pablito clavo un palito")
    assertEquals(tail_annotation.last.result, "un clavito chiquitillo")
  }



  "A mapAnnotationsCol" should "transform 2 documents columns to lower case" in {
    import SparkAccessor.spark.implicits._
    val df  = Seq(
      ("Pablito clavo un palito","Tres tristes tigres"),
      ("Un clavito chiquitillo","Comian trigo en un trigal")
    ).toDS.toDF("text","text2")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
    val documentAssembler2 = new DocumentAssembler()
      .setInputCol("text2")
      .setOutputCol("document2")
    val df2Documents =  documentAssembler.transform(df)

    val lower = (annotations: Seq[Annotation]) => {
      annotations.map(an => an.copy(result = an.result.toLowerCase))
    }

    import functions._
    val lowerDf = documentAssembler2.transform(df2Documents).mapAnnotationsCol[Seq[Annotation]](Seq("document","document2"),"tail_document","document",lower)
    val tail_annotation = Annotation.collect(lowerDf, "tail_document").flatten.toSeq.sortBy(_.begin)
    assertEquals(tail_annotation.head.result, "pablito clavo un palito")
    assertEquals(tail_annotation(1).result, "tres tristes tigres")
    assertEquals(tail_annotation(2).result, "un clavito chiquitillo")
    assertEquals(tail_annotation.last.result,"comian trigo en un trigal")
  }


}
