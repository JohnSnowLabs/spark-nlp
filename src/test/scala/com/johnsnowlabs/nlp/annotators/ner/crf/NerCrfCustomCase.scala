package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp.annotator.PerceptronModel
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsFormat}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher, LightPipeline, RecursivePipeline}
import org.apache.spark.ml.PipelineModel
import org.scalatest._

class NerCrfCustomCase extends FlatSpec {

  val spark = ResourceHelper.spark

  import spark.implicits._

  "NerCRF" should "read low trained model" ignore {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val pos = PerceptronModel.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val embeddings = new WordEmbeddings()
      .setInputCols("pos", "token", "sentence")
      .setOutputCol("embeddings")
      .setEmbeddingsSource("/emb.bin", 200, WordEmbeddingsFormat.BINARY)

    val nerCrf = new NerCrfApproach()
      .setInputCols("pos", "token", "sentence", "embeddings")
      .setOutputCol("ner")
      .setMinEpochs(50)
      .setMaxEpochs(80)
      .setLabelColumn("label")

    val finisher = new Finisher()
      .setInputCols("ner")

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        pos,
        embeddings,
        nerCrf,
        finisher
      ))

    val model = recursivePipeline.fit(Seq.empty[String].toDF("text"))

    model.write.overwrite().save("./crfnerconll")
    model.stages(4).asInstanceOf[NerCrfModel].write.overwrite().save("./crfnerconll-single")

  }

  "NerCRF" should "read and predict" ignore {
    val lp = new LightPipeline(PipelineModel.load("./crfnerconll"))

    println(lp.annotate(
      "Lung, right lower lobe, lobectomy: Grade 3"
    ))

  }

}
