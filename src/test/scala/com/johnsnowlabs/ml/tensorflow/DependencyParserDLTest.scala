/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.tensorflow

import org.scalatest.FlatSpec
import org.tensorflow.types.{TFloat32, TString}
import org.tensorflow.{AutoCloseableList, SavedModelBundle, Session, Tensor}

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

  def getOutputNextLstm(session: Session): AutoCloseableList[Tensor[_]] = {
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
