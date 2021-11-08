package com.johnsnowlabs.ml.pytorch

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import com.johnsnowlabs.nlp.util.io.ResourceHelper

import java.io.ByteArrayInputStream

class PytorchWrapper(pyTorchModel: Option[PtModel], modelBytes: Option[Array[Byte]]) extends Serializable {

  lazy val translator: Translator[Float, Float] = new Translator[Float, Float]() {
    override def processInput(ctx: TranslatorContext, input: Float): NDList = {
      val manager = ctx.getNDManager
      val array = manager.create(Array[Float](input))
      new NDList(array)
    }

    override def processOutput(ctx: TranslatorContext, list: NDList): Float = {
      val temp_arr = list.get(0)
      temp_arr.getFloat()
    }

    override def getBatchifier: Batchifier = { // The Batchifier describes how to combine a batch together
      // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
      Batchifier.STACK
    }
  }

  def infer(input: Float): Float = {

    println("************* Before predictor")
    val predictor = pyTorchModel.get.newPredictor(translator)
    println("************* Before predict")
    val output = predictor.predict(input)
    println(f"************* Before output: $output")
    output
  }

  def inferFromBytes(input: Float): Float = {
    println("************ In inferFromBytes")
    val pyTorchModel: PtModel = Model.newInstance("myPyTorchModel").asInstanceOf[PtModel]
    val modelInputStream = new ByteArrayInputStream(modelBytes.get)
    pyTorchModel.load(modelInputStream)

    println("************* Before predictor")
    val predictor = pyTorchModel.newPredictor(translator)
    println("************* Before predict")
    val output = predictor.predict(input)
    println(f"************* Before output: $output")
    output

  }

}

object PytorchWrapper extends Serializable {

  //TODO: Next run, try with Array[Byte]
  def read(pyTorchModelPath: String): PtModel = {
    val sourceStream = ResourceHelper.SourceStream(pyTorchModelPath)
    val pyTorchModel = Model.newInstance("myPyTorchModel").asInstanceOf[PtModel]
    pyTorchModel.load(sourceStream.pipe.head)
    pyTorchModel
  }

  def readBytes(pyTorchModelPath: String): Array[Byte] = {
    val sourceStream = ResourceHelper.SourceStream(pyTorchModelPath)
    val sourceBytes = new Array[Byte](sourceStream.pipe.head.available())

    sourceBytes
  }

}
