package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.ml.tensorflow.sentencepiece.LoadSentencepiece
import com.johnsnowlabs.nlp.annotators.ner.dl.LoadsContrib
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{FileHelper, ZipArchiveUtil}
import org.apache.commons.io.FileUtils
import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.hadoop.fs.Path
import org.slf4j.{Logger, LoggerFactory}
import org.tensorflow._
import org.tensorflow.exceptions.TensorFlowException
import org.tensorflow.proto.framework.{ConfigProto, GraphDef, TensorInfo}

import java.io._
import java.net.URI
import java.nio.file.{Files, Paths}
import java.util.{Map, UUID}
import scala.util.{Failure, Success, Try}


case class Variables(variables: Array[Byte], index: Array[Byte])


class TensorflowWrapper(var variables: Variables, var graph: Array[Byte]) extends Serializable {

  /** For Deserialization */
  def this() = {
    this(null, null)
  }

  @transient private var m_session: Session = _
  @transient private val logger = LoggerFactory.getLogger("TensorflowWrapper")

  def getSession(configProtoBytes: Option[Array[Byte]] = None): Session = {

    if (m_session == null){
      logger.debug("Restoring TF session from bytes")
      val t = new TensorResources()
      val config = configProtoBytes.getOrElse(TensorflowWrapper.TFSessionConfig)

      // save the binary data of variables to file - variables per se
      val path = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + TensorflowWrapper.TFVarsSuffix)
      val folder = path.toAbsolutePath.toString

      val varData = Paths.get(folder, TensorflowWrapper.VariablesPathValue)
      Files.write(varData, variables.variables)

      // save the binary data of variables to file - variables' index
      val varIdx = Paths.get(folder, TensorflowWrapper.VariablesIdxValue)
      Files.write(varIdx, variables.index)

      LoadsContrib.loadContribToTensorflow()

      // import the graph
      val _graph = new Graph()
      _graph.importGraphDef(GraphDef.parseFrom(graph))

      // create the session and load the variables
      val session = new Session(_graph, ConfigProto.parseFrom(config))
      val variablesPath = Paths.get(folder, TensorflowWrapper.VariablesKey).toAbsolutePath.toString

      session.runner.addTarget(TensorflowWrapper.SaveRestoreAllOP)
        .feed(TensorflowWrapper.SaveConstOP, t.createTensor(variablesPath))
        .run()

      //delete variable files
      Files.delete(varData)
      Files.delete(varIdx)

      m_session = session
    }
    m_session
  }

  def getTFHubSession(configProtoBytes: Option[Array[Byte]] = None, initAllTables: Boolean = true, loadSP: Boolean = false): Session = {

    if (m_session == null){
      logger.debug("Restoring TF Hub session from bytes")
      val t = new TensorResources()
      val config = configProtoBytes.getOrElse(TensorflowWrapper.TFSessionConfig)

      // save the binary data of variables to file - variables per se
      val path = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + TensorflowWrapper.TFVarsSuffix)
      val folder  = path.toAbsolutePath.toString
      val varData = Paths.get(folder, TensorflowWrapper.VariablesPathValue)
      Files.write(varData, variables.variables)

      // save the binary data of variables to file - variables' index
      val varIdx = Paths.get(folder, TensorflowWrapper.VariablesIdxValue)
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
      val session = new Session(g, ConfigProto.parseFrom(TensorflowWrapper.TFSessionConfig))
      val variablesPath = Paths.get(folder, TensorflowWrapper.VariablesKey).toAbsolutePath.toString
      if(initAllTables) {
        session.runner
          .addTarget(TensorflowWrapper.SaveRestoreAllOP)
          .addTarget(TensorflowWrapper.InitAllTableOP)
          .feed(TensorflowWrapper.SaveConstOP, t.createTensor(variablesPath))
          .run()
      }else{
        session.runner
          .addTarget(TensorflowWrapper.SaveRestoreAllOP)
          .feed(TensorflowWrapper.SaveConstOP, t.createTensor(variablesPath))
          .run()
      }

      //delete variable files
      Files.delete(varData)
      Files.delete(varIdx)

      m_session = session
    }
    m_session
  }

  def createSession(configProtoBytes: Option[Array[Byte]] = None): Session = {

    if (m_session == null){
      logger.debug("Creating empty TF session")

      val config = configProtoBytes.getOrElse(TensorflowWrapper.TFSessionConfig)

      LoadsContrib.loadContribToTensorflow()

      // import the graph
      val g = new Graph()
      g.importGraphDef(GraphDef.parseFrom(graph))

      // create the session and load the variables
      val session = new Session(g, ConfigProto.parseFrom(config))

      m_session = session
    }
    m_session
  }

  def saveToFile(file: String, configProtoBytes: Option[Array[Byte]] = None): Unit = {
    val t = new TensorResources()

    // 1. Create tmp director
    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    val variablesFile = Paths.get(folder, TensorflowWrapper.VariablesKey).toString

    // 2. Save variables
    getSession(configProtoBytes).runner.addTarget(TensorflowWrapper.SaveControlDependenciesOP)
      .feed(TensorflowWrapper.SaveConstOP, t.createTensor(variablesFile))
      .run()

    // 3. Save Graph
    // val graphDef = graph.toGraphDef
    val graphFile = Paths.get(folder, TensorflowWrapper.SavedModelPB).toString
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

    val variablesFile = Paths.get(folder, TensorflowWrapper.VariablesKey).toString

    // 2. Save variables
    getSession(configProtoBytes).runner.addTarget(TensorflowWrapper.SaveControlDependenciesOP)
      .feed(TensorflowWrapper.SaveConstOP, t.createTensor(variablesFile))
      .run()

    // 3. Save Graph
    val graphFile = Paths.get(folder, TensorflowWrapper.SavedModelPB).toString
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

      val varData = Paths.get(folder, TensorflowWrapper.VariablesPathValue)
      Files.write(varData, variables.variables)

      val varIdx = Paths.get(folder, TensorflowWrapper.VariablesIdxValue)
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

    this.m_session = tf.getSession()
    this.graph = tf.graph

    // 3. Delete tmp file
    FileHelper.delete(file.toAbsolutePath.toString)
  }
}

/** Companion object */
object TensorflowWrapper {
  private[TensorflowWrapper] val logger: Logger = LoggerFactory.getLogger("TensorflowWrapper")

  /** log_device_placement=True, allow_soft_placement=True, gpu_options.allow_growth=True */
  private final val TFSessionConfig: Array[Byte] = Array[Byte](50, 2, 32, 1, 56, 1)

  // Variables
  val VariablesKey = "variables"
  val VariablesPathValue = "variables.data-00000-of-00001"
  val VariablesIdxValue = "variables.index"

  // Operations
  val InitAllTableOP = "init_all_tables"
  val SaveRestoreAllOP = "save/restore_all"
  val SaveConstOP = "save/Const"
  val SaveControlDependenciesOP = "save/control_dependency"

  // Model
  val SavedModelPB = "saved_model.pb"

  // TF vars suffix folder
  val TFVarsSuffix = "_tf_vars"

  /** Utility method to load the TF saved model bundle */
  private def withSafeSavedModelBundleLoader(tags: Array[String], folder: String) = {
    import collection.JavaConverters._

    val model: SavedModelBundle =
      Try(SavedModelBundle.load(folder, tags: _*)) match {
        case Success(bundle) => bundle
        case Failure(s) => throw new Exception(s"Could not retrieve the SavedModelBundle + ${s.printStackTrace()}")
      }

    if (model.metaGraphDef.hasGraphDef && model.metaGraphDef.getSignatureDefCount > 0) {
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        val inputs: Map[String, TensorInfo] = sigDef.getInputsMap
        for (e <- inputs.entrySet.asScala) {
          val key: String = e.getKey
          val tfInfo: TensorInfo = e.getValue
          System.out.println("\nSignatureDef InputMap key: " + key + "\nSignatureDef InputMap tfInfo: " + tfInfo.getName)
        }
      }
      for (sigDef <- model.metaGraphDef.getSignatureDefMap.values.asScala) {
        val outputs: Map[String, TensorInfo] = sigDef.getOutputsMap
        for (e <- outputs.entrySet.asScala) {
          val key: String = e.getKey
          val tfInfo: TensorInfo = e.getValue
          System.out.println("\nSignatureDef OutputMap key: " + key + "\nSignatureDef OutputMap tfInfo: " + tfInfo.getName)
        }
      }
    }

    model
  }

  /** Utility method to load the TF saved model components without a provided bundle */
  private def unpackWithoutBundle(folder: String) = {
    val graph = readGraph(Paths.get(folder, SavedModelPB).toString)
    val session = new Session(graph, ConfigProto.parseFrom(TFSessionConfig))
    val varPath = Paths.get(folder, VariablesPathValue)
    val idxPath = Paths.get(folder, VariablesIdxValue)
    (graph, session, varPath, idxPath)
  }

  /** Utility method to load the TF saved model components from a provided bundle */
  private def unpackFromBundle(folder: String, model: SavedModelBundle) = {
    val graph = model.graph()
    val session = model.session()
    val varPath = Paths.get(folder, VariablesKey, VariablesPathValue)
    val idxPath = Paths.get(folder, VariablesKey, VariablesIdxValue)
    (graph, session, varPath, idxPath)
  }

  /** Utility method to process init all table operation key */
  private def processInitAllTableOp(initAllTables: Boolean,
                                    tensorResources: TensorResources,
                                    session: Session,
                                    variablesDir: String,
                                    variablesKey: String = VariablesKey) = {
    if (initAllTables) {
      session.runner
        .addTarget(SaveRestoreAllOP)
        .addTarget(InitAllTableOP)
        .feed(SaveConstOP, tensorResources.createTensor(Paths.get(variablesDir, variablesKey).toString))
        .run()
    } else {
      session.runner
        .addTarget(SaveRestoreAllOP)
        .feed(SaveConstOP, tensorResources.createTensor(Paths.get(variablesDir, variablesKey).toString))
        .run()
    }
  }

  /** Utility method to load a Graph from path */
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

  /**
    * Read method to create tmp folder, unpack archive and read file as SavedModelBundle
    * @param file: the file to read
    * @param zipped: boolean flag to know if compression is applied
    * @param useBundle: whether to use the SaveModelBundle object to parse the TF saved model
    * @param tags: tags to retrieve on the model bundle
    * @param initAllTables: boolean flag whether to retrieve the TF init operation
    * @return Returns a greeting based on the `name` field.
    */
  def read(file: String,
           zipped: Boolean = true,
           useBundle: Boolean = false,
           tags: Array[String] = Array.empty[String],
           initAllTables: Boolean = false)
  : TensorflowWrapper = {

    val t = new TensorResources()

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ner")
      .toAbsolutePath.toString

    // 2. Unpack archive
    val folder =
      if (zipped)
        ZipArchiveUtil.unzip(new File(file), Some(tmpFolder))
      else
        file

    LoadsContrib.loadContribToTensorflow()

    // 3. Read file as SavedModelBundle
    val (graph, session, varPath, idxPath) =
      if (useBundle) {
        val model: SavedModelBundle = withSafeSavedModelBundleLoader(tags = tags, folder = folder)
        val (graph, session, varPath, idxPath) = unpackFromBundle(folder, model)
        if(initAllTables) session.runner().addTarget(InitAllTableOP)
        (graph, session, varPath, idxPath)
      } else {
        val (graph, session, varPath, idxPath) = unpackWithoutBundle(folder)
        processInitAllTableOp(initAllTables, t, session, folder)
        (graph, session, varPath, idxPath)
      }

    val varBytes = Files.readAllBytes(varPath)
    val idxBytes = Files.readAllBytes(idxPath)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()
    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
    tfWrapper.m_session = session
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
    val (graph, session, varPath, idxPath) =
      if (useBundle) {
        val model: SavedModelBundle = withSafeSavedModelBundleLoader(tags = tags, folder = folder)
        val (graph, session, varPath, idxPath) = unpackFromBundle(folder, model)
        if(initAllTables) session.runner().addTarget(InitAllTableOP)
        (graph, session, varPath, idxPath)
      } else {
        val (graph, session, varPath, idxPath) = unpackWithoutBundle(folder)
        processInitAllTableOp(initAllTables, t, session, folder)
        (graph, session, varPath, idxPath)
      }

    val varBytes = Files.readAllBytes(varPath)
    val idxBytes = Files.readAllBytes(idxPath)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()

    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
    tfWrapper.m_session = session
    tfWrapper
  }

  def readZippedSavedModel(
                            rootDir: String = "",
                            fileName: String = "",
                            tags: Array[String] = Array.empty[String],
                            initAllTables: Boolean = false
                          ): TensorflowWrapper = {
    val tensorResources = new TensorResources()

    val listFiles = ResourceHelper.listResourceDirectory(rootDir)
    val path = if(listFiles.length > 1)
      s"${listFiles.head.split("/").head}/${fileName}"
    else listFiles.head

    val uri = new URI(path.replaceAllLiterally("\\", "/"))

    val inputStream = ResourceHelper.getResourceStream(uri.toString)

    // 1. Create tmp folder
    val tmpFolder =
      Files
        .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_classifier_dl_zip")
        .toAbsolutePath.toString

    val zipFile = new File(tmpFolder,"tmp_classifier_dl.zip")

    Files.copy(inputStream, zipFile.toPath)

    // 2. Unpack archive
    val folder = ZipArchiveUtil.unzip(zipFile, Some(tmpFolder))

    // 3. Create second tmp folder
    val finalFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_classifier_dl")
      .toAbsolutePath.toString

    val variablesFile = Paths.get(finalFolder, VariablesKey).toAbsolutePath
    Files.createDirectory(variablesFile)

    // 4. Copy the saved_model.zip into tmp folder
    val savedModelInputStream = ResourceHelper.getResourceStream(new Path(folder, SavedModelPB).toString)
    val savedModelFile = new File(finalFolder, SavedModelPB)
    Files.copy(savedModelInputStream, savedModelFile.toPath)

    val varIndexInputStream = ResourceHelper.getResourceStream(new Path(folder, VariablesIdxValue).toString)
    val varIndexFile = new File(variablesFile.toString, VariablesIdxValue)
    Files.copy(varIndexInputStream, varIndexFile.toPath)

    val varDataInputStream = ResourceHelper.getResourceStream(new Path(folder, VariablesPathValue).toString)
    val varDataFile = new File(variablesFile.toString, VariablesPathValue)
    Files.copy(varDataInputStream, varDataFile.toPath)

    // 5. Read file as SavedModelBundle
    val model = withSafeSavedModelBundleLoader(tags = tags, folder = finalFolder)

    val (graph, session, varPath, idxPath) = unpackFromBundle(finalFolder, model)

    if(initAllTables) session.runner().addTarget(InitAllTableOP)

    val varBytes = Files.readAllBytes(varPath)
    val idxBytes = Files.readAllBytes(idxPath)

    // 6. Remove tmp folder
    FileHelper.delete(tmpFolder)
    FileHelper.delete(finalFolder)
    FileHelper.delete(folder)
    tensorResources.clearTensors()

    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
    tfWrapper.m_session = session
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
    val folder =
      if (zipped)
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
    val variablesData = tfChkPointsVars(2).toString
    val variablesIndex = tfChkPointsVars(3).toString

    // 3. Read file as SavedModelBundle
    val graph = readGraph(Paths.get(folder, SavedModelPB).toString)
    val session = new Session(graph, ConfigProto.parseFrom(TFSessionConfig))
    val varPath = Paths.get(variablesData)
    val idxPath = Paths.get(variablesIndex)

    processInitAllTableOp(initAllTables, t, session, variablesDir, variablesKey = "part-00000-of-00001")

    val varBytes = Files.readAllBytes(varPath)
    val idxBytes = Files.readAllBytes(idxPath)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
    t.clearTensors()

    val tfWrapper = new TensorflowWrapper(Variables(varBytes, idxBytes), graph.toGraphDef.toByteArray)
    tfWrapper.m_session = session
    tfWrapper
  }

  def extractVariablesSavedModel(session: Session): Variables = {
    val t = new TensorResources()

    val folder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + TFVarsSuffix)
      .toAbsolutePath.toString
    val variablesFile = Paths.get(folder, VariablesKey).toString

    session.runner.addTarget(SaveControlDependenciesOP)
      .feed(SaveConstOP, t.createTensor(variablesFile))
      .run()

    val varPath = Paths.get(folder, VariablesPathValue)
    val varBytes = Files.readAllBytes(varPath)

    val idxPath = Paths.get(folder, VariablesIdxValue)
    val idxBytes = Files.readAllBytes(idxPath)

    val vars = Variables(varBytes, idxBytes)

    FileHelper.delete(folder)

    vars
  }

}
