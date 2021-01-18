package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest._


class MarianTransformerTestSpec extends FlatSpec {

  "MarianTransformer" should "correctly load pretrained model" ignore {
    import ResourceHelper.spark.implicits._

    val smallCorpus = Seq(
      "What is the capital of France?",
      "This should go to French",
      "This is a sentence in English that we want to translate to French",
      "Despite a Democratic majority in the General Assembly, Nunn was able to enact most of his priorities, including tax increases that funded improvements to the state park system and the construction of a statewide network of mental health centers."
    ).toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
      .setInputCols("document")
      .setOutputCol("sentence")

    val marian = MarianTransformer.pretrained()
      .setInputCols("sentence")
      .setOutputCol("translation")
      .setMaxInputLength(30)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        marian
      ))

    val pipelineModel = pipeline.fit(smallCorpus)

    Benchmark.time("Time to show") {
      pipelineModel.transform(smallCorpus).select("translation").show(false)
    }
    Benchmark.time("Time to second show") {
      pipelineModel.transform(smallCorpus).select("translation").show(false)
    }
    Benchmark.time("Time to save pipelineMolde") {
      pipelineModel.write.overwrite.save("./tmp_marianmt")
    }

    val savedPipelineModel = Benchmark.time("Time to load pipelineMolde") {
      PipelineModel.load("./tmp_marianmt")
    }
    val pipelineDF = Benchmark.time("Time to transform") {
      savedPipelineModel.transform(smallCorpus)
    }

    Benchmark.time("Time to show") {
      pipelineDF.select("translation").show(false)
    }
    Benchmark.time("Time to second show") {
      pipelineDF.select("translation").show(false)
    }
    Benchmark.time("Time to save pipeline") {
      pipelineModel.transform(smallCorpus).select("translation.result").write.mode("overwrite").save("./tmp_marianmt_pipeline")
    }
  }

}
