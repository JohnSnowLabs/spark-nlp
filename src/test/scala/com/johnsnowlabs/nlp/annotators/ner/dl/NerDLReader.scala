package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.LightPipeline
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.pretrained.pipelines.en.BasicPipeline
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest._

class NerDLReaderTestSpec extends FlatSpec {

  "Tensorflow NerDLReader" should "correctly load and save a ner model" ignore {

    val reader = NerDLModelPythonReader.read(
      "/conll_model/",
      ResourceHelper.spark,
      WordEmbeddingsFormat.TEXT
    )
    reader.write.overwrite().save("./nerconll")

    succeed
  }


  "NerDLModel" should "correctly read and use a tensorflow originated ner model" ignore {
    val spark = ResourceHelper.spark
    import spark.implicits._

    val bp = BasicPipeline().pretrained()

    val ner = NerDLModel.load("./nertst").setInputCols("document", "token").setOutputCol("ner")

    val np = new Pipeline().setStages(Array(bp, ner))

    val target = Array(
      "chronic obstructive pulmonary disease exacerbation",
      "acute hypertensive nephropathy",
      "moderate to severe enlargement of the cardiac silhouette",
      "An ultrasound of the right upper quadrant did not reveal any cholelithiasis or cholecystitis",
      "However , she has no vomiting and only mild nausea with medications .")

    val r = new LightPipeline(np.fit(Seq.empty[String].toDF("text")))
      .annotate(target)

    println(r.map(_.filterKeys(k => k == "document" || k == "ner")).mkString(","))

    succeed

  }

}
