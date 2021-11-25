package com.johnsnowlabs.ml.pytorch

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.pytorch.engine.PtModel
import ai.djl.translate.{Batchifier, Translator, TranslatorContext}

import java.io.ByteArrayInputStream

class PytorchLinearRegression(pytorchWrapper: PytorchWrapper) extends Serializable with Translator[Float, Float] {

  def inferFromBytes(input: Float): Float = {
    println("************ In inferFromBytes")
    val modelInputStream = new ByteArrayInputStream(pytorchWrapper.modelBytes)
    val pyTorchModel: PtModel = Model.newInstance("myPyTorchModel").asInstanceOf[PtModel]
    pyTorchModel.load(modelInputStream)

    println("************* Before predictor")
//    val translator = getTranslator
    val predictor = pyTorchModel.newPredictor(this)
    println("************* Before predict")
    val output: Float = predictor.predict(input)
    println(f"************* Before output: $output")
    output
  }

  override def getBatchifier: Batchifier = {
    // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
    Batchifier.STACK
  }

  override def processInput(ctx: TranslatorContext, input: Float): NDList = {
    val manager = ctx.getNDManager
    val array = manager.create(Array[Float](input))
    new NDList(array)
  }

  override def processOutput(ctx: TranslatorContext, list: NDList): Float = {
    val temp_arr = list.get(0)
    temp_arr.getFloat()
  }

  //  def getTranslator: Translator[Float, Float] = {
  //    new Translator[Float, Float]() {
  //      override def processInput(ctx: TranslatorContext, input: Float): NDList = {
  //        val manager = ctx.getNDManager
  //        val array = manager.create(Array[Float](input))
  //        new NDList(array)
  //      }
  //
  //      override def processOutput(ctx: TranslatorContext, list: NDList): Float = {
  //        val temp_arr = list.get(0)
  //        temp_arr.getFloat()
  //      }
  //
  //      override def getBatchifier: Batchifier = { // The Batchifier describes how to combine a batch together
  //        // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
  //        Batchifier.STACK
  //      }
  //    }
  //  }

}
