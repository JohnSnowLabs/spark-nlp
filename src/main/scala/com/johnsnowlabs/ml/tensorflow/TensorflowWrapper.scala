package com.johnsnowlabs.ml.tensorflow

import java.io.{File, IOException, ObjectOutputStream}
import java.nio.file.attribute.BasicFileAttributeView
import java.nio.file.{Files, Paths}
import java.util.UUID
import org.apache.commons.io.FileUtils
import org.tensorflow.{Graph, Session}


class TensorflowWrapper
(
  var session: Session,
  var graph: Graph
)  extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  private def getCreationTime(file: File): Long = {
    val path = Paths.get(file.getAbsolutePath())
    val view = Files.getFileAttributeView(path, classOf[BasicFileAttributeView]).readAttributes()
    view.creationTime().toMillis
  }

  def saveToFile(file: String): Unit = {
    val t = new TensorResources()

    // 1. Create tmp director
    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    val variablesFile = Paths.get(folder, "variables").toString

    // 2. Save variables
    session.runner.addTarget("save/control_dependency")
      .feed("save/Const", t.createTensor(variablesFile))
      .run()

    // 3. Save Graph
    val graphDef = graph.toGraphDef
    val graphFile = Paths.get(folder, "saved_model.pb").toString
    FileUtils.writeByteArrayToFile(new File(graphFile), graphDef)

    // 4. Zip folder
    ZipArchiveUtil.zip(folder, file)

    // 5. Remove tmp directory
    FileUtils.deleteDirectory(new File(folder))
    t.clearTensors()
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

  def read(file: String, zipped: Boolean = true): TensorflowWrapper = {
    val t = new TensorResources()

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    // 2. Unpack archive
    val folder = if (zipped)
      ZipArchiveUtil.unzip(new File(file), Some(tmpFolder))
    else
      file


    // 3. Read file as SavedModelBundle
    val graphDef = Files.readAllBytes(Paths.get(folder, "saved_model.pb"))
    val graph = new Graph()
    graph.importGraphDef(graphDef)
    val session = new Session(graph)
    session.runner.addTarget("save/restore_all")
      .feed("save/Const", t.createTensor(Paths.get(folder, "variables").toString))
      .run()

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
    t.clearTensors()

    new TensorflowWrapper(session, graph)
  }
}
