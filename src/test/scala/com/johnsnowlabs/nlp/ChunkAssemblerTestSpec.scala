package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest._

class ChunkAssemblerTestSpec extends FlatSpec {

  "a chunk assembler" should "correctly chunk ranges" in {
    import ResourceHelper.spark.implicits._

    val sampleDataset = Seq[(String, String)](
      ("Hello world, this is a sentence out of nowhere", "a sentence out"),
      ("Hey there, there is no chunk here", ""),
      ("Woah here, don't go so fast", "this is not there")
    ).toDF("sentence", "target")

    val answer = Array(
      Seq[Annotation](Annotation(AnnotatorType.CHUNK, 21, 34, "a sentence out", Map())),
      Seq.empty[Annotation],
      Seq.empty[Annotation]
    )

    val documentAssembler = new DocumentAssembler().setInputCol("sentence").setOutputCol("document")

    val chunkAssembler = new ChunkAssembler().setInputCols("document").setChunkCol("target").setOutputCol("chunk")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler))

    val results = pipeline.fit(Seq.empty[(String, String)].toDF("sentence", "target"))
      .transform(sampleDataset)
      .select( "chunk")
      .as[Seq[Annotation]]
        .collect()

    assert(results.zip(answer).forall(a => a._1.equals(a._2)))

  }

  "a chunk assembler" should "correctly chunk array ranges" in {
    import ResourceHelper.spark.implicits._

    val sampleDataset = Seq[(String, Seq[String])](
      ("Hello world, this is a sentence out of nowhere", Seq("world", "out of nowhere")),
      ("Hey there, there is no chunk here", Seq.empty[String]),
      ("Woah here, don't go so fast", Seq[String]("this is not there", "so fast"))
    ).toDF("sentence", "target")

    val answer = Array(
      Seq[Annotation](
        Annotation(AnnotatorType.CHUNK, 6, 10, "world", Map()),
        Annotation(AnnotatorType.CHUNK, 32, 45, "out of nowhere", Map())
      ),
      Seq.empty[Annotation],
      Seq[Annotation](
        Annotation(AnnotatorType.CHUNK, 20, 26, "so fast", Map())
      )
    )

    val documentAssembler = new DocumentAssembler().setInputCol("sentence").setOutputCol("document")

    val chunkAssembler = new ChunkAssembler().setIsArray(true).setInputCols("document").setChunkCol("target").setOutputCol("chunk")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler))

    val results = pipeline.fit(Seq.empty[(String, Seq[String])].toDF("sentence", "target"))
      .transform(sampleDataset)
      .select( "chunk")
      .as[Seq[Annotation]]
      .collect()

    assert(results.zip(answer).forall(a => a._1.equals(a._2)), s"\nbecause results were ${results.map(_.mkString(", ")).mkString(", ")}")

  }

}
