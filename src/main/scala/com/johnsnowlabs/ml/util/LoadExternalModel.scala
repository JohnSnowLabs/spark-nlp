/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.ml.util

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}

import java.io.File

object LoadExternalModel {

  def modelSanityCheck(modelPath: String): String = {

    /** Check if the path is correct */
    val f = new File(modelPath)

    require(f.exists, s"Folder $modelPath not found")
    require(f.isDirectory, s"File $modelPath is not folder")

    /*Check if the assets path is correct*/
    val assetsPath = modelPath + "/assets"

    require(f.exists, s"Folder $assetsPath not found")
    require(f.isDirectory, s"File $assetsPath is not folder")

    val tfSavedModel = new File(modelPath, ModelEngine.tensorflowModelName)
    val tfSavedModelExist = tfSavedModel.exists()

    val onnxModel = new File(modelPath, ModelEngine.onnxModelName)
    val onnxModelExist = onnxModel.exists()

    if (tfSavedModelExist) {
      ModelEngine.tensorflow
    } else if (onnxModelExist) {
      ModelEngine.onnx
    } else {
      // TODO: change this error once there is more than one DL engine
      require(
        tfSavedModelExist || onnxModelExist,
        s"Could not find saved_model.pb for TensorFlow model in $modelPath. Please make sure you" +
          s"followed provided notebooks to import TensorFlow models into Spark NLP: " +
          s"https://github.com/JohnSnowLabs/spark-nlp/discussions/5669")
      ModelEngine.unk
    }

  }

  def loadTextAsset(assetPath: String, assetName: String): Map[String, Int] = {

    val assetsPath = assetPath + "/assets"
    val assetFile = new File(assetsPath, assetName)
    require(assetFile.exists(), s"File $assetName not found in folder $assetsPath")

    val assetResource =
      new ExternalResource(assetFile.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    ResourceHelper.parseLines(assetResource).zipWithIndex.toMap

  }

}
