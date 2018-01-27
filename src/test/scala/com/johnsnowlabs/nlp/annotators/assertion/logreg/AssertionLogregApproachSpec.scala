package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.FlatSpec


class AssertionLogregApproachSpec extends FlatSpec {
  // load sample negex dataset
  val negexDataset = DataBuilder.loadParquetDataset("src/test/resources/negex.parquet")
  val logregPipelineModel = AnnotatorBuilder.getAssertionLogregModel(negexDataset)

  "AssertionLogregApproach" should "be serializable and deserializable correctly" in {
    logregPipelineModel.write.overwrite.save("./test_assertion_pipeline")
    val loadedAssertionPipeline = PipelineModel.read.load("./test_assertion_pipeline")
    val predicted = loadedAssertionPipeline.transform(negexDataset)

    assert(negexDataset.count == predicted.count)

  }

  "AssertionLogregApproach" should "have correct set of labels" in {
    val model = logregPipelineModel.stages(1).asInstanceOf[AssertionLogRegModel]

    assert(model.labelMap.get.get.size == 2)
    assert(model.labelMap.get.get.contains(1.0))
    assert(model.labelMap.get.get.contains(0.0))
    assert(model.labelMap.get.get.values.toList.contains("Affirmed"))
    assert(model.labelMap.get.get.values.toList.contains("Negated"))
  }


  "AssertionLogregApproach" should "produce meaningful assertions" in {
    val predicted = logregPipelineModel.transform(negexDataset)

    val annotations = Annotation.collect(predicted, "assertion").flatten.map(_.result).toSet

    assert(annotations.size == 2)
    assert(annotations.contains("Affirmed"))
    assert(annotations.contains("Negated"))

  }

}