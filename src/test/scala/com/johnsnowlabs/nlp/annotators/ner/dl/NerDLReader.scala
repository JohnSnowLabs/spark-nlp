package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.LightPipeline
import com.johnsnowlabs.nlp.annotators.NormalizerModel
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest._

class NerDLReaderTestSpec extends FlatSpec {

  "Tensorflow NerDLReader" should "correctly load and save a ner model" ignore {

    val model = NerDLModelPythonReader.read(
      "./source_model",
      ResourceHelper.spark,
      normalize = true,
      WordEmbeddingsFormat.BINARY
    )
    model.write.overwrite().save("./some_model")

    succeed
  }


  "NerDLModel" should "correctly read and use a tensorflow originated ner model" ignore {
    val spark = ResourceHelper.spark
    import spark.implicits._

    val bp = PretrainedPipeline("pipeline_basic")

    bp.model.stages(2).asInstanceOf[NormalizerModel]

    val ner = NerDLModel.load("./some_model").setInputCols("document", "normal").setOutputCol("ner")

    val np = new Pipeline().setStages(Array(bp.model, ner))

    val target = Array(
      "With regard to the patient's chronic obstructive pulmonary disease, the patient's respiratory status improved throughout the remainder of her hospital course.")

    val fit = np.fit(Seq.empty[String].toDF("text"))

    val r = new LightPipeline(fit)
      .annotate(target)

    println(r.map(_.filterKeys(k => k == "document" || k == "ner")).mkString(","))

    succeed

  }

}
