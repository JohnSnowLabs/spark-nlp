package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.LabeledDependency.DependencyInfo
import com.johnsnowlabs.nlp.annotators.common.{LabeledDependency, NerTagged}
import com.johnsnowlabs.nlp.annotators.ner.NerTagsEncoding
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{AveragedPerceptron, PerceptronModel}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.GraphBuilder
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.immutable.Map

class GraphExtraction(override val uid: String) extends AnnotatorModel[GraphExtraction]
  with HasSimpleAnnotate[GraphExtraction] {

  def this() = this(Identifiable.randomUID("GRAPH_EXTRACTOR"))

  protected val relationshipTypes = new StringArrayParam(this, "relationshipTypes",
    "Find paths between a pair of token and entity")

  protected val entityTypes = new StringArrayParam(this, "entityTypes",
    "Find paths between a pair of entities")

  protected val explodeEntities = new BooleanParam(this, "explodeEntities",
    "When set to true find paths between entities")

  protected val rootTokens = new StringArrayParam(this, "rootTokens",
    "Tokens to be consider as root to start traversing the paths. Use it along with explodeEntities")

  protected val maxSentenceSize = new IntParam(this, "maxSentenceSize",
    "Maximum sentence size that the annotator will process. Above this, the sentence is skipped")

  protected val minSentenceSize = new IntParam(this, "minSentenceSize",
    "Minimum sentence size that the annotator will process. Below this, the sentence is skipped")

  protected val mergeEntities = new BooleanParam(this, "mergeEntities",
    "Merge same neighboring entities as a single token")

  protected val includeEdges = new BooleanParam(this, "includeEdges",
    "Whether to include edges when building paths")

  protected val delimiter = new Param[String](this, "delimiter",
    "Delimiter symbol used for path output")

  protected val posModel = new StructFeature[PerceptronModel](this, "posModel")

//  protected val posModel = new Param[PerceptronModel](this, "posModel", "posModel")

  protected val dependencyParserModel: StructFeature[DependencyParserModel] =
    new StructFeature[DependencyParserModel](this, "dependencyParserModel")

  protected val typedDependencyParserModel: StructFeature[TypedDependencyParserModel] =
    new StructFeature[TypedDependencyParserModel](this, "typedDependencyParserModel")

  def setRelationshipTypes (value: Array[String]): this.type = set(relationshipTypes, value)

  def setEntityTypes(value: Array[String]): this.type = set(entityTypes, value)

  def setExplodeEntities(value: Boolean): this.type = set(explodeEntities, value)

  def setRootTokens(value: Array[String]): this.type  = set(rootTokens, value)

  def setMaxSentenceSize(value: Int): this.type = set(maxSentenceSize, value)

  def setMinSentenceSize(value: Int): this.type = set(minSentenceSize, value)

  def setMergeEntities(value: Boolean): this.type = set(mergeEntities, value)

  def setIncludeEdges(value: Boolean): this.type = set(includeEdges, value)

  def setDelimiter(value: String): this.type = set(delimiter, value)

//  def setPosModel(value: PerceptronModel): this.type = set(posModel, value)
//
//  def setDependencyParserModel(value: DependencyParserModel): this.type = set(dependencyParserModel, value)
//
//  def setTypedDependencyParserModel(value: TypedDependencyParserModel): this.type = set(typedDependencyParserModel, value)

  setDefault(entityTypes -> Array(), explodeEntities -> false, maxSentenceSize -> 1000, minSentenceSize -> 2,
    mergeEntities -> false, rootTokens -> Array(), relationshipTypes -> Array(), includeEdges -> true,
    delimiter -> ",")

  private lazy val allowedEntityRelationships = $(entityTypes).map{ entityRelationship =>
    val result = entityRelationship.split("-")
    (result.head, result.last)
  }.distinct

  private lazy val allowedRelationshipTypes = $(relationshipTypes).map{ relationshipTypes =>
    val result = relationshipTypes.split("-")
    (result.head, result.last)
  }.distinct

  private lazy val posPretrainedModel: Option[PerceptronModel] = if (posModel.isSet) Some(posModel.getOrDefault) else None

  private lazy val dependencyParserPretrainedModel: Option[DependencyParserModel] =
    if (dependencyParserModel.isSet) Some(dependencyParserModel.getOrDefault) else None

  private lazy val typedDependencyParserPretrainedModel: Option[TypedDependencyParserModel] =
    if (typedDependencyParserModel.isSet) Some(typedDependencyParserModel.getOrDefault) else None

  override def _transform(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DataFrame = {
    if ($(mergeEntities)) {
      super._transform(dataset, recursivePipeline)
    } else {
      val structFields = dataset.schema.fields
        .filter(field => field.metadata.contains("annotatorType"))
        .filter(field => field.metadata.getString("annotatorType") == DEPENDENCY ||
          field.metadata.getString("annotatorType") == LABELED_DEPENDENCY)
      if (structFields.length < 2 ) {
        throw new IllegalArgumentException(s"Missing either $DEPENDENCY or $LABELED_DEPENDENCY annotators. " +
          s"Make sure such annotators exist in your pipeline or setMergeEntities parameter to True")
      }

      val columnNames = structFields.map(structField => structField.name)
      val inputCols = getInputCols ++ columnNames
      val processedDataset = dataset.withColumn(
        getOutputCol, wrapColumnMetadata(dfAnnotate(array(inputCols.map(c => dataset.col(c)): _*))))
      processedDataset
    }
  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    //TODO: Add parameter to output path starting from root or bottom
    val sentenceIndexesToSkip = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)
      .filter(annotation => annotation.result.length > $(maxSentenceSize) || annotation.result.length < $(minSentenceSize))
      .map(annotation => annotation.metadata("sentence")).toList.distinct

    val annotationsToProcess = annotations.filter(annotation =>
      !sentenceIndexesToSkip.contains(annotation.metadata("sentence")))

    if (annotationsToProcess.isEmpty) {
      Seq(Annotation(NODE, 0, 0, "", Map()))
    } else {
      computeAnnotatePaths(annotationsToProcess)
    }
  }

  private def computeAnnotatePaths(annotations: Seq[Annotation]): Seq[Annotation] = {
    val annotationsBySentence = annotations.groupBy(token => token.metadata("sentence").toInt)
      .toSeq.sortBy(_._1)
      .map(annotationBySentence => annotationBySentence._2)

    val graphPaths = annotationsBySentence.flatMap{ sentenceAnnotations =>

      val annotations = if ($(mergeEntities)) getPretrainedAnnotations(sentenceAnnotations) else getEntityAnnotations(sentenceAnnotations)
      val tokens = annotations.filter(_.annotatorType == AnnotatorType.TOKEN)
      val nerEntities = annotations.filter(annotation =>
        annotation.annotatorType == TOKEN && annotation.metadata("entity") != "O")

      if (nerEntities.isEmpty) {
        Seq(Annotation(NODE, 0, 0, "", Map()))
      } else {
        val dependencyData = LabeledDependency.unpackHeadAndRelation(annotations)
        val annotationsInfo = AnnotationsInfo(tokens, nerEntities, dependencyData)

        val graph = new GraphBuilder(dependencyData.length + 1)
        dependencyData.zipWithIndex.foreach { case (dependencyInfo, index) =>
          graph.addEdge(dependencyInfo.headIndex, index + 1)
        }

        if ($(explodeEntities)) {
          extractGraphsFromEntities(annotationsInfo, graph)
        } else {
          extractGraphsFromRelationships(annotationsInfo, graph)
        }
      }
    }

    graphPaths

  }

  private def getPretrainedAnnotations(annotationsToProcess: Seq[Annotation]): Seq[Annotation] = {
    val relatedAnnotatedTokens = mergeRelatedTokens(annotationsToProcess)
    val sentence = annotationsToProcess.filter(_.annotatorType == AnnotatorType.DOCUMENT)
    val posInput = sentence ++ relatedAnnotatedTokens
    val posAnnotations = PretrainedAnnotations.getPos(posInput, posPretrainedModel)
    val dependencyParserInput = sentence ++ relatedAnnotatedTokens ++ posAnnotations
    val dependencyParserAnnotations = PretrainedAnnotations.getDependencyParser(dependencyParserInput,
      dependencyParserPretrainedModel)
    val typedDependencyParserInput = relatedAnnotatedTokens ++ posAnnotations ++ dependencyParserAnnotations
    val typedDependencyParserAnnotations = PretrainedAnnotations.getTypedDependencyParser(typedDependencyParserInput,
      typedDependencyParserPretrainedModel)
    relatedAnnotatedTokens ++ dependencyParserAnnotations ++ typedDependencyParserAnnotations
  }

  private def getEntityAnnotations(annotationsToProcess: Seq[Annotation]): Seq[Annotation] = {
    val entityAnnotations = annotationsToProcess.filter(_.annotatorType == NAMED_ENTITY)
    val tokensWithEntity = annotationsToProcess.filter(_.annotatorType == TOKEN).zipWithIndex.map{
      case (annotation, index) =>
        val tag = entityAnnotations(index).result
        val entity = if (tag.length == 1) tag else tag.substring(2)
        val metadata = annotation.metadata ++ Map("entity" -> entity)
        Annotation(annotation.annotatorType, annotation.begin, annotation.end, annotation.result, metadata)
     }
    val dependencyParserAnnotations = annotationsToProcess.filter(annotation =>
      annotation.annotatorType == DEPENDENCY || annotation.annotatorType == LABELED_DEPENDENCY)

    tokensWithEntity ++ dependencyParserAnnotations
  }

  private def mergeRelatedTokens(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = NerTagged.unpack(annotations)
    val docs = annotations.filter(a => a.annotatorType == AnnotatorType.DOCUMENT && sentences.exists(
      b => b.indexedTaggedWords.exists(c => c.begin >= a.begin && c.end <= a.end)
    ))

    val entities = sentences.zip(docs.zipWithIndex).flatMap { case (sentence, doc) =>
      NerTagsEncoding.fromIOB(sentence, doc._1, sentenceIndex = doc._2, includeNoneEntities = true)
    }

    entities.map(entity =>
      Annotation(TOKEN, entity.start, entity.end, entity.text,
        Map("sentence" -> entity.sentenceId, "entity" -> entity.entity))
    )
  }

  private def extractGraphsFromEntities(annotationsInfo: AnnotationsInfo, graph: GraphBuilder): Seq[Annotation] = {

    var rootIndices: Array[Int] = Array()

    if ($(rootTokens).isEmpty) {
      val sourceDependency = annotationsInfo.dependencyData.filter(dependencyInfo => dependencyInfo.headIndex == 0)
      rootIndices = Array(annotationsInfo.dependencyData.indexOf(sourceDependency.head) + 1)
    } else {
      val sourceDependencies = $(rootTokens).flatMap(rootToken =>
        annotationsInfo.dependencyData.filter(_.token == rootToken))
      rootIndices = sourceDependencies.map(sourceDependency =>
        annotationsInfo.dependencyData.indexOf(sourceDependency) + 1)
    }

    val entitiesPairData = getEntitiesData(annotationsInfo.nerEntities, annotationsInfo.dependencyData)
    val annotatedPaths = rootIndices.flatMap(rootIndex =>
      getAnnotatedPaths(entitiesPairData, graph, rootIndex, annotationsInfo.tokens))
    annotatedPaths
  }

  private def extractGraphsFromRelationships(annotationsInfo: AnnotationsInfo, graph: GraphBuilder): Seq[Annotation] = {

    val annotatedGraphPaths = allowedRelationshipTypes.flatMap{ relationshipTypes =>
      val rootData = annotationsInfo.tokens.filter(_.result == relationshipTypes._1).map(token =>
        (token, annotationsInfo.tokens.indexOf(token) + 1))
      val entityIndexes = annotationsInfo.nerEntities.filter(_.metadata("entity") == relationshipTypes._2)
        .map(nerEntity => annotationsInfo.tokens.indexOf(nerEntity) + 1)

      rootData.flatMap{ rootInfo =>
        val paths = entityIndexes.flatMap(entityIndex =>
          buildPath(graph, (rootInfo._2, entityIndex), annotationsInfo.dependencyData))
        val pathsMap = paths.zipWithIndex.flatMap{ case (path, index) => Map(s"path${(index + 1).toString}" -> path)}.toMap
        if (paths.nonEmpty) {
          Some(Annotation(NODE, rootInfo._1.begin, rootInfo._1.end, rootInfo._1.result,
            Map("relationship" -> s"${rootInfo._1.result},${relationshipTypes._2}") ++ pathsMap)
          )
        } else {
          None
        }
      }
    }
    annotatedGraphPaths
  }

  private def buildPath(graph: GraphBuilder, nodesIndexes: (Int, Int), dependencyData: Seq[DependencyInfo]):
  Option[String] = {
    val nodesIndexesPath = graph.findPath(nodesIndexes._1, nodesIndexes._2)
    val path = nodesIndexesPath.map{ nodeIndex =>
      val dependencyInfo =  dependencyData(nodeIndex - 1)
      var result = dependencyInfo.token
      if ($(includeEdges)) {
        val edge = if (dependencyInfo.relation == "*root*") "" else dependencyInfo.relation + $(delimiter)
        result = edge + dependencyInfo.token
      }
      result
    }
    if (path.isEmpty) None else Some(path.mkString($(delimiter)))
  }

  private def getAnnotatedPaths(entitiesPairData: List[EntitiesPairInfo], graph: GraphBuilder,
                                rootIndex: Int, tokens: Seq[Annotation]): Seq[Annotation] = {

    val paths = entitiesPairData.flatMap{ entitiesPairInfo =>
      val leftPath = graph.findPath(rootIndex, entitiesPairInfo.entitiesIndex._1)
      val rightPath = graph.findPath(rootIndex, entitiesPairInfo.entitiesIndex._2)
      if (leftPath.nonEmpty && rightPath.nonEmpty) {
        Some(GraphInfo(entitiesPairInfo.entities, leftPath, rightPath))
      } else {
        None
      }
    }

    val sourceToken = tokens(rootIndex - 1)
    val annotatedPaths = paths.map{ path =>
      val leftEntity = path.entities._1
      val rightEntity = path.entities._2
      val leftPathTokens = path.leftPathIndex.map(index => tokens(index - 1).result)
      val rightPathTokens = path.rightPathIndex.map(index => tokens(index - 1).result)
      val fullPath = leftPathTokens.mkString($(delimiter)) + $(delimiter) + rightPathTokens.mkString($(delimiter))

      Annotation(NODE, sourceToken.begin, sourceToken.end, sourceToken.result,
        Map("entities" -> s"$leftEntity,$rightEntity", "path" -> fullPath,
          "left_path" -> leftPathTokens.mkString($(delimiter)), "right_path" -> rightPathTokens.mkString($(delimiter)))
      )
    }
    annotatedPaths
  }

  private def getEntitiesData(annotatedEntities: Seq[Annotation], dependencyData: Seq[DependencyInfo]):
  List[EntitiesPairInfo] = {
    var annotatedEntitiesPairs: List[(Annotation, Annotation)] = List()
    if (allowedEntityRelationships.isEmpty) {
      annotatedEntitiesPairs = annotatedEntities.combinations(2).map(entity => (entity.head, entity.last)).toList
    } else {
      annotatedEntitiesPairs = allowedEntityRelationships.flatMap(entities =>
        getAnnotatedNerEntitiesPairs(entities, annotatedEntities))
        .filter(entities => entities._1.begin != entities._2.begin && entities._1.end != entities._2.end)
        .toList
    }

    val entitiesPairData = annotatedEntitiesPairs.map{ annotatedEntityPair =>
      val dependencyInfoLeft = dependencyData.filter(dependencyInfo =>
        dependencyInfo.beginToken == annotatedEntityPair._1.begin && dependencyInfo.endToken == annotatedEntityPair._1.end
      )
      val dependencyInfoRight = dependencyData.filter(dependencyInfo =>
        dependencyInfo.beginToken == annotatedEntityPair._2.begin && dependencyInfo.endToken == annotatedEntityPair._2.end
      )
      val indexLeft = dependencyData.indexOf(dependencyInfoLeft.head) + 1
      val indexRight = dependencyData.indexOf(dependencyInfoRight.head) + 1

      EntitiesPairInfo((indexLeft, indexRight),
        (annotatedEntityPair._1.metadata("entity"), annotatedEntityPair._2.metadata("entity")))
    }
    entitiesPairData.distinct
  }

  private def getAnnotatedNerEntitiesPairs(entities: (String, String), annotatedEntities: Seq[Annotation]):
  List[(Annotation, Annotation)] = {

    val leftEntities = annotatedEntities.filter(annotatedEntity => annotatedEntity.metadata("entity") == entities._1)
    val rightEntities = annotatedEntities.filter(annotatedEntity => annotatedEntity.metadata("entity") == entities._2)

    if (leftEntities.length > rightEntities.length) {
      leftEntities.flatMap{ leftEntity => rightEntities.map(rightEntity => (leftEntity, rightEntity))}.toList
    } else {
      rightEntities.flatMap{ rightEntity => leftEntities.map(leftEntity => (leftEntity, rightEntity))}.toList
    }

  }

  private case class EntitiesPairInfo(entitiesIndex: (Int, Int), entities: (String, String))
  private case class GraphInfo(entities: (String, String), leftPathIndex: List[Int], rightPathIndex: List[Int])
  private case class AnnotationsInfo(tokens: Seq[Annotation], nerEntities: Seq[Annotation],
                                     dependencyData: Seq[DependencyInfo])

  override val outputAnnotatorType: AnnotatorType = NODE
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, NAMED_ENTITY)
}
