package com.johnsnowlabs.nlp.annotators.multipleannotations

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.{ContentProvider, DocumentAssembler, LightPipeline, RecursivePipeline, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.{TextMatcher, Tokenizer}
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class MultiannotationsSpec  extends AnyFlatSpec {
  import SparkAccessor.spark.implicits._

  "An multiple anootator chunks" should "transform data " taggedAs FastTest in {
    val data = SparkAccessor.spark.sparkContext.parallelize(Seq("Example text")).toDS().toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val documentAssembler2 = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document2")

    val documentAssembler3 = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document3")

    val multipleColumns = new MultiColumnApproach().setInputCols("document","document2","document3").setOutputCol("merge")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        documentAssembler2,
        documentAssembler3,
        multipleColumns
      ))

    val pipelineModel = pipeline.fit(data)

    pipelineModel.transform(data).show(truncate = false)

    val result = new LightPipeline(pipelineModel).annotate("My document")

    println(result)

  }



}
