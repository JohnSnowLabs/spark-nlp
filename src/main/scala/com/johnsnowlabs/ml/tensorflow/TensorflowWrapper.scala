package com.johnsnowlabs.ml.tensorflow

import java.io._
import java.nio.file.Files
import java.util.UUID

import com.johnsnowlabs.nlp.annotators.ner.dl.LoadsContrib
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow._
import java.nio.file.Paths


case class Variables(variables:Array[Byte], index:Array[Byte])
class TensorflowWrapper(
  var variables: Variables,
  var graph: Array[Byte]
)  extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  @transient private var msession: Session = _
  @transient private val logger = LoggerFactory.getLogger("TensorflowWrapper")

  def getSession(loadsContrib: Boolean = false): Session = {

    if (msession == null){
      logger.debug("Restoring TF session from bytes")
      val t = new TensorResources()
      val config = Array[Byte](50, 2, 32, 1, 56, 1, 64, 1)

      // save the binary data of variables to file - variables per se
      val path = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_tf_vars")
      val folder  = path.toAbsolutePath.toString
      val varData = Paths.get(folder, "variables.data-00000-of-00001")
      Files.write(varData, variables.variables)

      // save the binary data of variables to file - variables' index
      val varIdx = Paths.get(folder, "variables.index")
      Files.write(varIdx, variables.index)

      if (loadsContrib)
        TensorflowWrapper.loadContribToTensorflow()

      // import the graph
      val g = new Graph()
      g.importGraphDef(graph)

      // create the session and load the variables
      val session = new Session(g, config)
      val variablesPath = Paths.get(folder, "variables").toAbsolutePath.toString
      session.runner.addTarget("save/restore_all")
        .feed("save/Const", t.createTensor(variablesPath))
        .run()

      //delete variable files
      Files.delete(varData)
      Files.delete(varIdx)

      msession = session
    }
    msession
  }

  def createSession(loadsContrib: Boolean = false): Session = {

    if (msession == null){
      logger.debug("Creating empty TF session")

      val config = Array[Byte](50, 2, 32, 1, 56, 1)

      if (loadsContrib)
        TensorflowWrapper.loadContribToTensorflow()

      // import the graph
      val g = new Graph()
      g.importGraphDef(graph)

      // create the session and load the variables
      val session = new Session(g, config)

      msession = session
    }
    msession
  }

  def saveToFile(file: String, loadsContrib: Boolean = false): Unit = {
    val t = new TensorResources()

    // 1. Create tmp director
    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    val variablesFile = Paths.get(folder, "variables").toString

    // 2. Save variables
    getSession(loadsContrib).runner.addTarget("save/control_dependency")
      .feed("save/Const", t.createTensor(variablesFile))
      .run()

    // 3. Save Graph
    // val graphDef = graph.toGraphDef
    val graphFile = Paths.get(folder, "saved_model.pb").toString
    FileUtils.writeByteArrayToFile(new File(graphFile), graph)

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

    this.msession = tf.getSession()
    this.graph = tf.graph

    // 3. Delete tmp file
    FileHelper.delete(file.toAbsolutePath.toString)
  }
}

object TensorflowWrapper extends LoadsContrib {
  private[TensorflowWrapper] val logger: Logger = LoggerFactory.getLogger("TensorflowWrapper")

  def readGraph(graphFile: String, loadContrib: Boolean = false): Graph = {
    if (loadContrib)
      loadContribToTensorflow()
    val graphBytesDef = FileUtils.readFileToByteArray(new File(graphFile))
    val graph = new Graph()
    graph.importGraphDef(graphBytesDef)
    graph
  }

  def read(
            file: String,
            zipped: Boolean = true,
            useBundle: Boolean = false,
            tags: Array[String] = Array.empty[String],
            loadContrib: Boolean = false
          ): TensorflowWrapper = {
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
    //val config = Array[Byte](50, 2, 32, 1, 56, 1, 64, 1)
    /** without log placement */
    val config = Array[Byte](50, 2, 32, 1, 56, 1)
    // val config = Array[Byte](56, 1)

    // 3. Read file as SavedModelBundle
    val (graph, session, varPath, idxPath) = if (useBundle) {
      val model = SavedModelBundle.load(folder, tags: _*)
      val graph = model.graph()
      val session = model.session()
      val varPath = Paths.get(folder, "variables", "variables.data-00000-of-00001")
      val idxPath = Paths.get(folder, "variables", "variables.index")
      (graph, session, varPath, idxPath)
    } else {
      val graph = readGraph(Paths.get(folder, "saved_model.pb").toString, loadContrib = loadContrib)
      val session = new Session(graph, config)
      val varPath = Paths.get(folder, "variables.data-00000-of-00001")
      val idxPath = Paths.get(folder, "variables.index")
      session.runner.addTarget("save/restore_all")
        .feed("save/Const", t.createTensor(Paths.get(folder, "variables").toString))
        .run()
      (graph, session, varPath, idxPath)
    }

    val varBytes = Files.readAllBytes(varPath)

    val idxBytes = Files.readAllBytes(idxPath)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()

    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef)
    tfWrapper.msession = session
    tfWrapper
  }

  def extractVariables(session: Session): Variables = {
    val t = new TensorResources()

    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_tf_vars")
      .toAbsolutePath.toString
    val variablesFile = Paths.get(folder, "variables").toString

    session.runner.addTarget("save/control_dependency")
      .feed("save/Const", t.createTensor(variablesFile))
      .run()

    val varPath = Paths.get(folder, "variables.data-00000-of-00001")
    val varBytes = Files.readAllBytes(varPath)

    val idxPath = Paths.get(folder, "variables.index")
    val idxBytes = Files.readAllBytes(idxPath)

    val vars = Variables(varBytes, idxBytes)

    FileHelper.delete(folder)

    vars
  }

}
