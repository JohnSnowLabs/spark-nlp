package com.johnsnowlabs.ml.tensorflow.wrap

import org.tensorflow.Session

trait TFWrapper[S <: TFWrapper[S]] {

  def getTFHubSession(configProtoBytes: Option[Array[Byte]] = None,
                      initAllTables: Boolean = true,
                      loadSP: Boolean = false,
                      savedSignatures: Option[Map[String, String]] = None): Session

  def getGraph(): Array[Byte]

  def saveToFile(tfFile: String, configProtoBytes: Option[Array[Byte]]): Unit

  def getSession(configProtoBytes: Option[Array[Byte]]): Session

  def createSession(configProtoBytes: Option[Array[Byte]]): Session

  def saveToFileV1V2(tfFile: String,
                     configProtoBytes: Option[Array[Byte]],
                     savedSignatures: Option[Map[String, String]]): Unit

  def useTFIO: Boolean

  def getUseTFIO: Boolean = useTFIO

  def read(file: String,
           zipped: Boolean = true,
           useBundle: Boolean = false,
           tags: Array[String] = Array.empty[String],
           initAllTables: Boolean = false,
           savedSignatures: Option[Map[String, String]] = None): (TFWrapper[_], Option[Map[String, String]])
}