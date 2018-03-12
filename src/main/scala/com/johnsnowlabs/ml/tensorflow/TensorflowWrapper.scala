package com.johnsnowlabs.ml.tensorflow

import java.io.{File, IOException, ObjectOutputStream}
import java.nio.file.{Files, Paths}
import java.util.UUID
import org.apache.commons.io.FileUtils
import org.tensorflow.{Graph, SavedModelBundle, Session}


class TensorflowWrapper
(
  var session: Session,
  var graph: Graph
)  extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  def saveToFile(file: String): Unit = {
    // 1. Create tmp director
    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    val modelName = "model_temp_e08d7ffea7a143eeabde4444e8d81456"

    // 2. Save variables
    session.runner.addTarget("save/SaveV2").run()
    val variablesFile = Paths.get(folder, "variables").toString
    FileUtils.moveDirectory(new File(modelName), new File(variablesFile))

    // 3. Save Graph
    val graphDef = graph.toGraphDef
    val graphFile = Paths.get(folder, "/saved_model.pb").toString
    FileUtils.writeByteArrayToFile(new File(graphFile), graphDef)

    // 4. Zip folder
    ZipArchiveUtil.zip(folder, file)

    // 5. Remove tmp directory
    FileUtils.deleteDirectory(new File(folder))
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    // 1. Create tmp file
    val file = Files.createTempFile("tf", "zip")

    // 2. save to file
    this.saveToFile(file.toString)

    // 3. Unpack and read TF state
    val result = TensorflowWrapper.read(file.toString)

    // 4. Copy params
    this.session = result.session
    this.graph = result.graph

    // 5. Remove tmp archive
    FileUtils.deleteQuietly(new File(file.toString))
  }
}

object TensorflowWrapper {
  def read(file: String): TensorflowWrapper = {
    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    // 2. Unpack archive
    val folder = ZipArchiveUtil.unzip(new File(file), Some(tmpFolder))

    // 3. Read file as SavedModelBundle
    val result = SavedModelBundle.load(folder)

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(folder))

    new TensorflowWrapper(result.session, result.graph)
  }
}
