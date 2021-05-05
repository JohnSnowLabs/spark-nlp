package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import org.scalatest.FlatSpec
import org.tensorflow.types.{TFloat32, TString}
import org.tensorflow.{SavedModelBundle, Session, Tensor}

class DependencyParserDLTest extends FlatSpec {

  "TensorFlow Wrapper" should "deserialize saved model" in {
    val tags: Array[String] = Array(SavedModelBundle.DEFAULT_TAG)
    val modelPath: String = "/home/danilo/IdeaProjects/JSL/bist-parser-tensorflow/model-small-tf/dp-parser.model7/BiLSTM/"
    val prefix: String = modelPath + "variables/variables"
    val model: SavedModelBundle = TensorflowWrapper.withSafeSavedModelBundleLoader(tags = tags, savedModelDir = modelPath)
    val session = model.session()
    val saverDef = model.metaGraphDef().getSaverDef

    session.runner()
      .addTarget(saverDef.getRestoreOpName)
      .feed(saverDef.getFilenameTensorName, TString.scalarOf(prefix))
      .run()

    val outputNextLstm = getOutputNextLstm(session)

    outputNextLstm.forEach{output =>
      val tensor: Tensor[TFloat32] = output.expect(TFloat32.DTYPE)
      println(tensor.toString)
    }
    outputNextLstm.close()
  }

  def getOutputNextLstm(session: Session) = {
    val wNextLstm = "bi_lstm_model/NextBlockLSTM/w_next_lstm/Read/ReadVariableOp"
    val wigNextLstm = "bi_lstm_model/NextBlockLSTM/wig_next_lstm/Read/ReadVariableOp"
    val wfgNextLstm = "bi_lstm_model/NextBlockLSTM/wfg_next_lstm/Read/ReadVariableOp"
    val wogNextLstm = "bi_lstm_model/NextBlockLSTM/wog_next_lstm/Read/ReadVariableOp"

    val outputSecondLstm = new AutoCloseableList[Tensor[_]](
      session.runner()
        .fetch(wNextLstm)
        .fetch(wigNextLstm)
        .fetch(wfgNextLstm)
        .fetch(wogNextLstm)
        .run()
    )
    outputSecondLstm
  }

}
