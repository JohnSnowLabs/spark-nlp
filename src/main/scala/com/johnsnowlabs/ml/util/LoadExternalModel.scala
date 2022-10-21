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
import java.net.{URI, URL}
import java.nio.file.Paths
import scala.io.Source

object LoadExternalModel {

  val notSupportedEngineError: String =
    "Your imported model is not supported. Please make sure you" +
      s"follow provided notebooks to import external models into Spark NLP: " +
      s"https://github.com/JohnSnowLabs/spark-nlp/discussions/5669"

  def modelSanityCheck(modelPath: String): String = {

    /** Check if the path is correct */
    val f = new File(modelPath)
    require(f.exists, s"Folder $modelPath not found")
    require(f.isDirectory, s"Folder $modelPath is not folder")

    /*Check if the assets path is correct*/
    val assetsPath = Paths.get(modelPath, "/assets").toString
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
      require(tfSavedModelExist || onnxModelExist, notSupportedEngineError)
      ModelEngine.unk
    }

  }

  /** Retrieves a local path for a model folder.
    *
    * If the model is at a remote location it will be downloaded and a local path provided.
    * Otherwise an URL to the local path of the folder will be returned.
    *
    * @param path
    *   Local or Remote path of the model folder
    * @return
    *   URL to the local path of the folder
    */
  def retrieveModel(path: String): (URL, String) = {

    val localFileUri: URI = {
      val localModelUri = ResourceHelper.copyToLocalSavedModel(path)

      // Get absolute path so file protocol is included
      if (Option(localModelUri.getScheme).isEmpty) Paths.get(localModelUri).toAbsolutePath.toUri
      else localModelUri
    }

    val localPath: String = localFileUri.getPath

    (localFileUri.toURL, modelSanityCheck(localPath))
  }

  def loadTextAsset(assetPath: String, assetName: String): Array[String] = {
    val assetFile = checkAndCreateFile(assetPath + "/assets", assetName)

    // Convert to URL first to access correct file protocol
    val assetResource =
      new ExternalResource(assetFile.toURI.toURL.toString, ReadAs.TEXT, Map("format" -> "text"))
    ResourceHelper.parseLines(assetResource)
  }

  /** @param assetPath
    *   path to root of assets directory
    * @param assetName
    *   asset's name
    * @return
    *   SentencePieceWrapper
    */
  def loadSentencePieceAsset(assetPath: String, assetName: String): SentencePieceWrapper = {
    val assetFile = checkAndCreateFile(assetPath + "/assets", assetName)
    SentencePieceWrapper.read(assetFile.toString)
  }

  /** @param assetPath
    *   path to root of assets directory
    * @param assetName
    *   asset's name
    * @return
    *   JSON as String to be parsed later
    */
  def loadJsonStringAsset(assetPath: String, assetName: String): String = {
    val assetFile = checkAndCreateFile(assetPath + "/assets", assetName)
    val vocabStream = ResourceHelper.getResourceStream(assetFile.getAbsolutePath)
    Source.fromInputStream(vocabStream).mkString
  }

  /** @param filePath
    *   path to the file
    * @param fileName
    *   file's name
    * @return
    *   File if the file exists
    */
  private def checkAndCreateFile(filePath: String, fileName: String): File = {
    val f = new File(filePath, fileName)
    require(f.exists(), s"File $fileName not found in folder $filePath")
    f
  }

}
