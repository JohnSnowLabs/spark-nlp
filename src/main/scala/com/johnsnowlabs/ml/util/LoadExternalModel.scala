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

import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}

import java.io.File

object LoadExternalModel {

  def modelSanityCheck(modelPath: String): String = {

    /** Check if the path is correct */
    val f = new File(modelPath)
    require(f.exists, s"Folder $modelPath not found")
    require(f.isDirectory, s"Folder $modelPath is not folder")

    /*Check if the assets path is correct*/
    val assetsPath = modelPath + "/assets"
    val assetsPathFile = new File(assetsPath)
    require(assetsPathFile.exists, s"Folder $assetsPath not found")
    require(assetsPathFile.isDirectory, s"Folder $assetsPath is not folder")

    /*TensorFlow required model's name*/
    val tfSavedModel = new File(modelPath, ModelEngine.tensorflowModelName)
    val tfSavedModelExist = tfSavedModel.exists()

    /*ONNX required model's name*/
    val onnxModel = new File(modelPath, ModelEngine.onnxModelName)
    val onnxModelExist = onnxModel.exists()

    if (tfSavedModelExist) {
      ModelEngine.tensorflow
    } else if (onnxModelExist) {
      ModelEngine.onnx
    } else {
      require(
        tfSavedModelExist || onnxModelExist,
        s"Could not find saved_model.pb for TensorFlow model in $modelPath. Please make sure you" +
          s"follow provided notebooks to import external models into Spark NLP: " +
          s"https://github.com/JohnSnowLabs/spark-nlp/discussions/5669")
      ModelEngine.unk
    }

  }

  def loadTextAsset(assetPath: String, assetName: String): Array[String] = {
    val assetFile = checkAndCreateFile(assetPath + "/assets", assetName)
    val assetResource =
      new ExternalResource(assetFile.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    ResourceHelper.parseLines(assetResource)
  }

  def loadSentencePieceAsset(assetPath: String, assetName: String): SentencePieceWrapper = {
    val assetFile = checkAndCreateFile(assetPath + "/assets", assetName)
    SentencePieceWrapper.read(assetFile.toString)
  }

  private def checkAndCreateFile(filePath: String, fileName: String): File = {
    val f = new File(filePath, fileName)
    require(f.exists(), s"File $fileName not found in folder $filePath")
    f
  }

}
