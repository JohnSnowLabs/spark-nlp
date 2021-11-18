package com.johnsnowlabs.ml.pytorch

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}
import com.johnsnowlabs.nlp.util.io.ResourceHelper

import java.io.ByteArrayInputStream

class PytorchWrapper(modelBytes: Option[Array[Byte]]) extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null) //TODO: Check if this is really required
  }

  def inferFromBytes(input: Float): Float = {
    val translator = getTranslator
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

  def getTranslator: Translator[Float, Float] = {
    new Translator[Float, Float]() {
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
  }

}

object PytorchWrapper {

  def read(pyTorchModelPath: String): PytorchWrapper = {
    val modelBytes = readBytes(pyTorchModelPath)
    new PytorchWrapper(Some(modelBytes))
  }

  def readBytes(pyTorchModelPath: String): Array[Byte] = {
    //TODO: Verify if we don't need to save model in a local tmp file and that stuff from TensorflowWrapper
    val sourceStream = ResourceHelper.SourceStream(pyTorchModelPath)
    val inputStreamModel = sourceStream.pipe.head
    val modelBytes = new Array[Byte](inputStreamModel.available())
    inputStreamModel.read(modelBytes)

    modelBytes
  }

}
