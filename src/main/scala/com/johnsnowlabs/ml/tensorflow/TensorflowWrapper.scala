package com.johnsnowlabs.ml.tensorflow

import java.io._
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.{FileUtils, IOUtils}
import org.apache.commons.lang.SystemUtils
import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow._
import org.tensorflow.TensorFlowException


class TensorflowWrapper(
  var session: Session,
  var graph: Graph
) extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
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
    FileHelper.delete(folder)
    t.clearTensors()
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    // 1. Create tmp file
    val file = Files.createTempFile("tf", "zip")

    // 2. save to file
    this.saveToFile(file.toString)

    // 3. Read state as bytes array
    val result = Files.readAllBytes(file)

    // 4. Save to out stream
    out.writeObject(result)

    // 5. Remove tmp archive
    FileHelper.delete(file.toAbsolutePath.toString)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    // 1. Create tmp file
    val file = Files.createTempFile("tf", "zip")
    val bytes = in.readObject().asInstanceOf[Array[Byte]]
    Files.write(file.toAbsolutePath, bytes)

    // 2. Read from file
    val tf = TensorflowWrapper.read(file.toString, true)
    this.session = tf.session
    this.graph = tf.graph

    // 3. Delete tmp file
    FileHelper.delete(file.toAbsolutePath.toString)
  }
}

object TensorflowWrapper {
  private[TensorflowWrapper] val logger: Logger = LoggerFactory.getLogger("TensorflowWrapper")

  def readGraph(graphFile: String, handleException: Boolean = true): Graph = {
    val graphStream = ResourceHelper.getResourceStream(graphFile)
    val graphBytesDef = if (graphStream != null)
      IOUtils.toByteArray(graphStream)
    else
      FileUtils.readFileToByteArray(new File(graphFile))

    val graph = new Graph()
    if (!handleException) {
      graph.importGraphDef(graphBytesDef)
      return graph
    }

    try {
      graph.importGraphDef(graphBytesDef)
      graph
    }
    catch {
      case _: TensorFlowException =>
        // trying to add library
        logger.info("Problem with loading graph. Trying to add .so library")
        val os = System.getProperty("os.name").toLowerCase()
        logger.debug("os name: "+os)
        val paths: Option[(String, String)] =
          if (SystemUtils.IS_OS_MAC || SystemUtils.IS_OS_MAC_OSX) {
            Some(("ner-dl/mac/_sparse_feature_cross_op.so", "ner-dl/mac/_lstm_ops.so"))
          } else if (SystemUtils.IS_OS_WINDOWS) {
            None
          } else {
            Some(("ner-dl/linux/_sparse_feature_cross_op.so", "ner-dl/linux/_lstm_ops.so"))
          }

        if (paths.isDefined) {

          val resource = ResourceHelper.copyResourceToTmp(paths.get._1)
          TensorFlow.loadLibrary(resource.getPath)
          resource.delete()

          val resource2 = ResourceHelper.copyResourceToTmp(paths.get._2)
          TensorFlow.loadLibrary(resource2.getPath)
          resource2.delete()

          logger.info("Added contrib .so library")
        } else {
          logger.info("Windows detected. Using noncontrib files")
          logger.warn("Using NON-contrib DL graphs due to Windows OS. Accuracy may be lower than optimal. (Fix me)")
        }

        val graph = readGraph(graphFile, handleException=false)

        logger.info("Graph loaded")

        graph
    }
  }

  def read(file: String, zipped: Boolean = true, useBundle: Boolean = false, tags: Array[String] = Array.empty[String]): TensorflowWrapper = {
    val t = new TensorResources()

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    // 2. Unpack archive
    val folder = if (zipped)
      ZipArchiveUtil.unzip(new File(file), Some(tmpFolder))
    else
      file

    //Use CPU
    //val config = Array[Byte](10, 7, 10, 3, 67, 80, 85, 16, 0)
    //Use GPU
    /** log_device_placement=True, allow_soft_placement=True, gpu_options.allow_growth=True*/
    val config = Array[Byte](50, 2, 32, 1, 56, 1, 64, 1)
    // val config = Array[Byte](56, 1)

    // 3. Read file as SavedModelBundle
    val (graph, session) = if (useBundle) {
      val model = SavedModelBundle.load(folder, tags: _*)
      val graph = model.graph()
      val session = model.session()
      (graph, session)
    } else {
      val graph = readGraph(Paths.get(folder, "saved_model.pb").toString)
      val session = new Session(graph, config)
      session.runner.addTarget("save/restore_all")
        .feed("save/Const", t.createTensor(Paths.get(folder, "variables").toString))
        .run()
      (graph, session)
    }

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()

    new TensorflowWrapper(session, graph)
  }
}