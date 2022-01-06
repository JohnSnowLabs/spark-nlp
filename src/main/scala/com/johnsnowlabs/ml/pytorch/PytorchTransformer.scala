package com.johnsnowlabs.ml.pytorch

import ai.djl.inference.Predictor
import ai.djl.ndarray.NDList
import ai.djl.{Device, Model}
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import com.johnsnowlabs.nlp.embeddings.TransformerEmbeddings

import java.io.ByteArrayInputStream

trait PytorchTransformer extends Translator[Array[Array[Int]], Array[Array[Float]]] with TransformerEmbeddings {

  val pytorchWrapper: PytorchWrapper

  protected lazy val predictor: Predictor[Array[Array[Int]], Array[Array[Float]]] = {
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)
    val device = Device.cpu() //TODO: Check with gpu
    val model = Model.newInstance("pytorch-model", device)

    val pyTorchModel: PtModel = model.asInstanceOf[PtModel]
    pyTorchModel.load(modelInputStream)

    pyTorchModel.newPredictor(this)
  }

  override def tag(batch: Seq[Array[Int]]): Seq[Array[Array[Float]]] = {

    val maxSentenceLength = batch.map(encodedSentence => encodedSentence.length).max
    val output = predictor.predict(batch.toArray)
    val dimension = output.head.head.toInt
    val allEncoderLayers = output.last
    val predictedEmbeddings = allEncoderLayers
      .grouped(dimension).toArray
      .grouped(maxSentenceLength).toArray

    val emptyVector = Array.fill(dimension)(0f)
    batch.zip(predictedEmbeddings).map { case (ids, embeddings) =>
      if (ids.length > embeddings.length) {
        embeddings.take(embeddings.length - 1) ++
          Array.fill(embeddings.length - ids.length)(emptyVector) ++
          Array(embeddings.last)
      } else {
        embeddings
      }
    }
  }

  override def getBatchifier: Batchifier = {
    Batchifier.fromString("none")
  }

  override def processInput(ctx: TranslatorContext, input: Array[Array[Int]]): NDList = {
    val manager = ctx.getNDManager
    val array = manager.create(input)
    new NDList(array)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Array[Array[Float]] = {

    val dimension = Array(list.get(0).getShape.get(2).toFloat)
    val allEncoderLayers = list.get(0).toFloatArray

    Array(dimension, allEncoderLayers)
  }

}
