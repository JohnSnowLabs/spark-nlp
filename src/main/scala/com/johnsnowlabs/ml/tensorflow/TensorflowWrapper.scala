package com.johnsnowlabs.ml.tensorflow

import java.io._
import java.nio.file.Files
import java.util.UUID

import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow._
import java.nio.file.Paths

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper.tfSessionConfig
import com.johnsnowlabs.ml.tensorflow.sentencepiece.LoadSentencepiece
import com.johnsnowlabs.nlp.annotators.ner.dl.LoadsContrib
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.hadoop.fs.Path
import org.tensorflow.exceptions.TensorFlowException
import org.tensorflow.proto.framework.{ConfigProto, GraphDef}


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

  /** log_device_placement=True, allow_soft_placement=True, gpu_options.allow_growth=True*/
  val tfSessionConfig: Array[Byte] = Array[Byte](50, 2, 32, 1, 56, 1)

  def getSession(configProtoBytes: Option[Array[Byte]] = None): Session = {

    if (msession == null){
      logger.debug("Restoring TF session from bytes")
      val t = new TensorResources()
      val config = configProtoBytes.getOrElse(tfSessionConfig)

      // save the binary data of variables to file - variables per se
      val path = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_tf_vars")
      val folder = path.toAbsolutePath.toString
      val varData = Paths.get(folder, "variables.data-00000-of-00001")
      Files.write(varData, variables.variables)

      // save the binary data of variables to file - variables' index
      val varIdx = Paths.get(folder, "variables.index")
      Files.write(varIdx, variables.index)

      LoadsContrib.loadContribToTensorflow()

      // import the graph
      val g = new Graph()
      g.importGraphDef(GraphDef.parseFrom(graph))

      // create the session and load the variables
      val session = new Session(g, ConfigProto.parseFrom(config))
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

  def getTFHubSession(configProtoBytes: Option[Array[Byte]] = None, initAllTables: Boolean = true, loadSP: Boolean = false): Session = {

    if (msession == null){
      logger.debug("Restoring TF Hub session from bytes")
      val t = new TensorResources()
      val config = configProtoBytes.getOrElse(tfSessionConfig)

      // save the binary data of variables to file - variables per se
      val path = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_tf_vars")
      val folder  = path.toAbsolutePath.toString
      val varData = Paths.get(folder, "variables.data-00000-of-00001")
      Files.write(varData, variables.variables)

      // save the binary data of variables to file - variables' index
      val varIdx = Paths.get(folder, "variables.index")
      Files.write(varIdx, variables.index)

      LoadsContrib.loadContribToTensorflow()
      if(loadSP) {
        LoadSentencepiece.loadSPToTensorflowLocally()
        LoadSentencepiece.loadSPToTensorflow()
      }
      // import the graph
      val g = new Graph()
      g.importGraphDef(GraphDef.parseFrom(graph))

      // create the session and load the variables
      val session = new Session(g, ConfigProto.parseFrom(tfSessionConfig))
      val variablesPath = Paths.get(folder, "variables").toAbsolutePath.toString
      if(initAllTables) {
        session.runner
          .addTarget("save/restore_all")
          .addTarget("init_all_tables")
          .feed("save/Const", t.createTensor(variablesPath))
          .run()
      }else{
        session.runner
          .addTarget("save/restore_all")
          .feed("save/Const", t.createTensor(variablesPath))
          .run()
      }

      //delete variable files
      Files.delete(varData)
      Files.delete(varIdx)

      msession = session
    }
    msession
  }

  def createSession(configProtoBytes: Option[Array[Byte]] = None): Session = {

    if (msession == null){
      logger.debug("Creating empty TF session")

      val config = configProtoBytes.getOrElse(tfSessionConfig)

      LoadsContrib.loadContribToTensorflow()

      // import the graph
      val g = new Graph()
      g.importGraphDef(GraphDef.parseFrom(graph))

      // create the session and load the variables
      val session = new Session(g, ConfigProto.parseFrom(config))

      msession = session
    }
    msession
  }

  def saveToFile(file: String, configProtoBytes: Option[Array[Byte]] = None): Unit = {
    val t = new TensorResources()

    // 1. Create tmp director
    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    val variablesFile = Paths.get(folder, "variables").toString

    // 2. Save variables
    getSession(configProtoBytes).runner.addTarget("save/control_dependency")
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
  /*
  * saveToFileV2 is V2 compatible
  * */
  def saveToFileV1V2(file: String, configProtoBytes: Option[Array[Byte]] = None): Unit = {
    val t = new TensorResources()

    // 1. Create tmp director
    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    val variablesFile = Paths.get(folder, "variables").toString

    // 2. Save variables
    getSession(configProtoBytes).runner.addTarget("save/control_dependency")
      .feed("save/Const", t.createTensor(variablesFile))
      .run()

    // 3. Save Graph
    val graphFile = Paths.get(folder, "saved_model.pb").toString
    FileUtils.writeByteArrayToFile(new File(graphFile), graph)

    val tfChkPointsVars = FileUtils.listFilesAndDirs(
      new File(folder),
      new WildcardFileFilter("part*"),
      new WildcardFileFilter("variables*")
    ).toArray()

    // TF2 Saved Model generate parts for variables on second save
    // This makes sure they are compatible with V1
    if(tfChkPointsVars.length > 3){
      val variablesDir = tfChkPointsVars(1).toString

      val varData = Paths.get(folder, "variables.data-00000-of-00001")
      Files.write(varData, variables.variables)

      val varIdx = Paths.get(folder, "variables.index")
      Files.write(varIdx, variables.index)

      FileHelper.delete(variablesDir)
    }

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
    val tf = TensorflowWrapper.read(file.toString, zipped = true)

    this.msession = tf.getSession()
    this.graph = tf.graph

    // 3. Delete tmp file
    FileHelper.delete(file.toAbsolutePath.toString)
  }
}

object TensorflowWrapper {
  private[TensorflowWrapper] val logger: Logger = LoggerFactory.getLogger("TensorflowWrapper")

  private val tfSessionConfig: Array[Byte] = Array[Byte](50, 2, 32, 1, 56, 1)

  def readGraph(graphFile: String): Graph = {
    val graphBytesDef = FileUtils.readFileToByteArray(new File(graphFile))
    val graph = new Graph()
    try {
      graph.importGraphDef(GraphDef.parseFrom(graphBytesDef))
    } catch {
      case e: TensorFlowException if e.getMessage.contains("Op type not registered 'BlockLSTM'") =>
        throw new UnsupportedOperationException("Spark NLP tried to load a TensorFlow Graph using Contrib module, but" +
          " failed to load it on this system. If you are on Windows, please follow the correct steps for setup: " +
          "https://github.com/JohnSnowLabs/spark-nlp/issues/1022" +
          s" If not the case, please report this issue. Original error message:\n\n${e.getMessage}")
    }
    graph
  }

  def read(
            file: String,
            zipped: Boolean = true,
            useBundle: Boolean = false,
            tags: Array[String] = Array.empty[String],
            initAllTables: Boolean = false
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

    LoadsContrib.loadContribToTensorflow()

    // 3. Read file as SavedModelBundle
    val (graph, session, varPath, idxPath) = if (useBundle) {
      val model = SavedModelBundle.load(folder, tags: _*)
      val graph = model.graph()
      val session = model.session()
      val varPath = Paths.get(folder, "variables", "variables.data-00000-of-00001")
      val idxPath = Paths.get(folder, "variables", "variables.index")
      if(initAllTables) {
        session.runner().addTarget("init_all_tables")
      }
      (graph, session, varPath, idxPath)
    } else {
      val graph = readGraph(Paths.get(folder, "saved_model.pb").toString)

      val session = new Session(graph, ConfigProto.parseFrom(tfSessionConfig))
      val varPath = Paths.get(folder, "variables.data-00000-of-00001")
      val idxPath = Paths.get(folder, "variables.index")
      if(initAllTables) {
        session.runner
          .addTarget("save/restore_all")
          .addTarget("init_all_tables")
          .feed("save/Const", t.createTensor(Paths.get(folder, "variables").toString))
          .run()
      }else{
        session.runner
          .addTarget("save/restore_all")
          .feed("save/Const", t.createTensor(Paths.get(folder, "variables").toString))
          .run()
      }
      (graph, session, varPath, idxPath)
    }

    val varBytes = Files.readAllBytes(varPath)

    val idxBytes = Files.readAllBytes(idxPath)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()
    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
    tfWrapper.msession = session
    tfWrapper
  }

  def readWithSP(
                  file: String,
                  zipped: Boolean = true,
                  useBundle: Boolean = false,
                  tags: Array[String] = Array.empty[String],
                  initAllTables: Boolean = false,
                  loadSP: Boolean = false
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

    if(loadSP) {
      LoadSentencepiece.loadSPToTensorflowLocally()
      LoadSentencepiece.loadSPToTensorflow()
    }
    // 3. Read file as SavedModelBundle
    val (graph, session, varPath, idxPath) = if (useBundle) {
      val model = SavedModelBundle.load(folder, tags: _*)
      val graph = model.graph()
      val session = model.session()
      val varPath = Paths.get(folder, "variables", "variables.data-00000-of-00001")
      val idxPath = Paths.get(folder, "variables", "variables.index")
      if(initAllTables) {
        session.runner().addTarget("init_all_tables")
      }
      (graph, session, varPath, idxPath)
    } else {
      val graph = readGraph(Paths.get(folder, "saved_model.pb").toString)
      val session = new Session(graph, ConfigProto.parseFrom(tfSessionConfig))
      val varPath = Paths.get(folder, "variables.data-00000-of-00001")
      val idxPath = Paths.get(folder, "variables.index")
      if(initAllTables) {
        session.runner
          .addTarget("save/restore_all")
          .addTarget("init_all_tables")
          .feed("save/Const", t.createTensor(Paths.get(folder, "variables").toString))
          .run()
      }else{
        session.runner
          .addTarget("save/restore_all")
          .feed("save/Const", t.createTensor(Paths.get(folder, "variables").toString))
          .run()
      }
      (graph, session, varPath, idxPath)
    }

    val varBytes = Files.readAllBytes(varPath)

    val idxBytes = Files.readAllBytes(idxPath)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()

    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
    tfWrapper.msession = session
    tfWrapper
  }

  def readZippedSavedModel(
                            rootDir: String = "",
                            fileName: String = "",
                            tags: Array[String] = Array.empty[String],
                            initAllTables: Boolean = false
                          ): TensorflowWrapper = {
    val t = new TensorResources()

    val listFiles = ResourceHelper.listResourceDirectory(rootDir)
    val path = if(listFiles.length > 1)
      s"${listFiles.head.split("/").head}/${fileName}"
    else listFiles.head

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))

    val inputStream = ResourceHelper.getResourceStream(uri.toString)

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_classifier_dl_zip")
      .toAbsolutePath.toString

    val zipFIle = new File(tmpFolder,"tmp_classifier_dl.zip")

    Files.copy(inputStream, zipFIle.toPath)

    // 2. Unpack archive
    val folder = ZipArchiveUtil.unzip(zipFIle, Some(tmpFolder))

    // 3. Create second tmp folder
    val finalFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_classifier_dl")
      .toAbsolutePath.toString

    val variablesFile = Paths.get(finalFolder, "variables").toAbsolutePath
    Files.createDirectory(variablesFile)

    // 4. Copy the saved_model.zip into tmp folder
    val savedModelInputStream = ResourceHelper.getResourceStream(new Path(folder, "saved_model.pb").toString)
    val savedModelFile = new File(finalFolder,"saved_model.pb")
    Files.copy(savedModelInputStream, savedModelFile.toPath)

    val varIndexInputStream = ResourceHelper.getResourceStream(new Path(folder, "variables.index").toString)
    val varIndexFile = new File(variablesFile.toString,"variables.index")
    Files.copy(varIndexInputStream, varIndexFile.toPath)

    val varDataInputStream = ResourceHelper.getResourceStream(new Path(folder, "variables.data-00000-of-00001").toString)
    val varDataFile = new File(variablesFile.toString,"variables.data-00000-of-00001")
    Files.copy(varDataInputStream, varDataFile.toPath)

    // 5. Read file as SavedModelBundle
    val model = SavedModelBundle.load(finalFolder, tags: _*)
    val graph = model.graph()
    val session = model.session()
    val varPath = Paths.get(finalFolder, "variables", "variables.data-00000-of-00001")
    val idxPath = Paths.get(finalFolder, "variables", "variables.index")
    if(initAllTables) {
      session.runner().addTarget("init_all_tables")
    }

    val varBytes = Files.readAllBytes(varPath)

    val idxBytes = Files.readAllBytes(idxPath)

    // 6. Remove tmp folder
    FileHelper.delete(tmpFolder)
    FileHelper.delete(finalFolder)
    FileHelper.delete(folder)
    t.clearTensors()

    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
    tfWrapper.msession = session
    tfWrapper
  }

  def readChkPoints(
                     file: String,
                     zipped: Boolean = true,
                     tags: Array[String] = Array.empty[String],
                     initAllTables: Boolean = false
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

    LoadsContrib.loadContribToTensorflow()

    val tfChkPointsVars = FileUtils.listFilesAndDirs(
      new File(folder),
      new WildcardFileFilter("part*"),
      new WildcardFileFilter("variables*")
    ).toArray()

    val variablesDir = tfChkPointsVars(1).toString
    val variablseData = tfChkPointsVars(2).toString
    val variablesIndex = tfChkPointsVars(3).toString

    // 3. Read file as SavedModelBundle
    val graph = readGraph(Paths.get(folder, "saved_model.pb").toString)

    val session = new Session(graph, ConfigProto.parseFrom(tfSessionConfig))
    val varPath = Paths.get(variablseData)
    val idxPath = Paths.get(variablesIndex)
    if(initAllTables) {
      session.runner
        .addTarget("save/restore_all")
        .addTarget("init_all_tables")
        .feed("save/Const", t.createTensor(Paths.get(variablesDir, "part-00000-of-00001").toString))
        .run()
    }else{
      session.runner
        .addTarget("save/restore_all")
        .feed("save/Const", t.createTensor(Paths.get(variablesDir, "part-00000-of-00001").toString))
        .run()
    }

    val varBytes = Files.readAllBytes(varPath)

    val idxBytes = Files.readAllBytes(idxPath)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()

    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
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

  def extractVariablesSavedModel(session: Session): Variables = {
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
