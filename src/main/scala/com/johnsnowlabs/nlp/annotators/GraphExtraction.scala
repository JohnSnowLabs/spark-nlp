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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.LabeledDependency.DependencyInfo
import com.johnsnowlabs.nlp.annotators.common.{LabeledDependency, NerTagged}
import com.johnsnowlabs.nlp.annotators.ner.NerTagsEncoding
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.util.GraphBuilder
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.{DataFrame, Dataset}

/** Extracts a dependency graph between entities.
  *
  * The GraphExtraction class takes e.g. extracted entities from a
  * [[com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel NerDLModel]] and creates a dependency tree
  * which describes how the entities relate to each other. For that a triple store format is used.
  * Nodes represent the entities and the edges represent the relations between those entities. The
  * graph can then be used to find relevant relationships between words.
  *
  * Both the
  * [[com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel DependencyParserModel]] and
  * [[com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel TypedDependencyParserModel]]
  * need to be present in the pipeline. There are two ways to set them:
  *
  *   1. Both Annotators are present in the pipeline already. The dependencies are taken
  *      implicitly from these two Annotators.
  *   1. Setting `setMergeEntities` to `true` will download the default pretrained models for
  *      those two Annotators automatically. The specific models can also be set with
  *      `setDependencyParserModel` and `setTypedDependencyParserModel`:
  *      {{{
  *            val graph_extraction = new GraphExtraction()
  *              .setInputCols("document", "token", "ner")
  *              .setOutputCol("graph")
  *              .setRelationshipTypes(Array("prefer-LOC"))
  *              .setMergeEntities(true)
  *            //.setDependencyParserModel(Array("dependency_conllu", "en",  "public/models"))
  *            //.setTypedDependencyParserModel(Array("dependency_typed_conllu", "en",  "public/models"))
  *      }}}
  *
  * To transform the resulting graph into a more generic form such as RDF, see the
  * [[com.johnsnowlabs.nlp.GraphFinisher GraphFinisher]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  * import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
  * import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
  * import org.apache.spark.ml.Pipeline
  * import com.johnsnowlabs.nlp.annotators.GraphExtraction
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("sentence")
  *   .setOutputCol("token")
  *
  * val embeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("embeddings")
  *
  * val nerTagger = NerDLModel.pretrained()
  *   .setInputCols("sentence", "token", "embeddings")
  *   .setOutputCol("ner")
  *
  * val posTagger = PerceptronModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("pos")
  *
  * val dependencyParser = DependencyParserModel.pretrained()
  *   .setInputCols("sentence", "pos", "token")
  *   .setOutputCol("dependency")
  *
  * val typedDependencyParser = TypedDependencyParserModel.pretrained()
  *   .setInputCols("dependency", "pos", "token")
  *   .setOutputCol("dependency_type")
  *
  * val graph_extraction = new GraphExtraction()
  *   .setInputCols("document", "token", "ner")
  *   .setOutputCol("graph")
  *   .setRelationshipTypes(Array("prefer-LOC"))
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   tokenizer,
  *   embeddings,
  *   nerTagger,
  *   posTagger,
  *   dependencyParser,
  *   typedDependencyParser,
  *   graph_extraction
  * ))
  *
  * val data = Seq("You and John prefer the morning flight through Denver").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("graph").show(false)
  * +-----------------------------------------------------------------------------------------------------------------+
  * |graph                                                                                                            |
  * +-----------------------------------------------------------------------------------------------------------------+
  * |[[node, 13, 18, prefer, [relationship -> prefer,LOC, path1 -> prefer,nsubj,morning,flat,flight,flat,Denver], []]]|
  * +-----------------------------------------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.GraphFinisher GraphFinisher]] to output the paths in a more generic
  *   format, like RDF
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class GraphExtraction(override val uid: String)
    extends AnnotatorModel[GraphExtraction]
    with HasSimpleAnnotate[GraphExtraction] {

  def this() = this(Identifiable.randomUID("GRAPH_EXTRACTOR"))

  /** Find paths between a pair of token and entity (Default: `Array()`)
    *
    * @group param
    */
  val relationshipTypes = new StringArrayParam(
    this,
    "relationshipTypes",
    "Find paths between a pair of token and entity")

  /** Find paths between a pair of entities (Default: `Array()`)
    *
    * @group param
    */
  val entityTypes =
    new StringArrayParam(this, "entityTypes", "Find paths between a pair of entities")

  /** When set to true find paths between entities (Default: `false`)
    *
    * @group param
    */
  val explodeEntities =
    new BooleanParam(this, "explodeEntities", "When set to true find paths between entities")

  /** Tokens to be consider as root to start traversing the paths (Default: `Array()`). Use it
    * along with `explodeEntities`
    *
    * @group param
    */
  val rootTokens = new StringArrayParam(
    this,
    "rootTokens",
    "Tokens to be consider as root to start traversing the paths. Use it along with explodeEntities")

  /** Maximum sentence size that the annotator will process (Default: `1000`). Above this, the
    * sentence is skipped
    *
    * @group param
    */
  val maxSentenceSize = new IntParam(
    this,
    "maxSentenceSize",
    "Maximum sentence size that the annotator will process. Above this, the sentence is skipped")

  /** Minimum sentence size that the annotator will process (Default: `2`). Below this, the
    * sentence is skipped
    *
    * @group param
    */
  val minSentenceSize = new IntParam(
    this,
    "minSentenceSize",
    "Minimum sentence size that the annotator will process. Below this, the sentence is skipped")

  /** Merge same neighboring entities as a single token (Default: `false`)
    *
    * @group param
    */
  val mergeEntities =
    new BooleanParam(this, "mergeEntities", "Merge same neighboring entities as a single token")

  /** IOB format to apply when merging entities
    *
    * @group param
    */
  val mergeEntitiesIOBFormat = new Param[String](
    this,
    "mergeEntitiesIOBFormat",
    "IOB format to apply when merging entities. Values: IOB or IOB2")

  /** Whether to include edges when building paths (Default: `true`)
    *
    * @group param
    */
  val includeEdges =
    new BooleanParam(this, "includeEdges", "Whether to include edges when building paths")

  /** Delimiter symbol used for path output (Default: `","`)
    *
    * @group param
    */
  val delimiter = new Param[String](this, "delimiter", "Delimiter symbol used for path output")

  /** Coordinates (name, lang, remoteLoc) to a pretrained POS model (Default: `Array()`)
    *
    * @group param
    */
  val posModel = new StringArrayParam(
    this,
    "posModel",
    "Coordinates (name, lang, remoteLoc) to a pretrained POS model")

  /** Coordinates (name, lang, remoteLoc) to a pretrained Dependency Parser model (Default:
    * `Array()`)
    *
    * @group param
    */
  val dependencyParserModel = new StringArrayParam(
    this,
    "dependencyParserModel",
    "Coordinates (name, lang, remoteLoc) to a pretrained Dependency Parser model")

  /** Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency Parser model (Default:
    * `Array()`)
    *
    * @group param
    */
  val typedDependencyParserModel = new StringArrayParam(
    this,
    "typedDependencyParserModel",
    "Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency Parser model")

  /** @group setParam */
  def setRelationshipTypes(value: Array[String]): this.type = set(relationshipTypes, value)

  /** @group setParam */
  def setEntityTypes(value: Array[String]): this.type = set(entityTypes, value)

  /** @group setParam */
  def setExplodeEntities(value: Boolean): this.type = set(explodeEntities, value)

  /** @group setParam */
  def setRootTokens(value: Array[String]): this.type = set(rootTokens, value)

  /** @group setParam */
  def setMaxSentenceSize(value: Int): this.type = set(maxSentenceSize, value)

  /** @group setParam */
  def setMinSentenceSize(value: Int): this.type = set(minSentenceSize, value)

  /** @group setParam */
  def setMergeEntities(value: Boolean): this.type = set(mergeEntities, value)

  /** @group setParam */
  def setMergeEntitiesIOBFormat(value: String): this.type = set(mergeEntitiesIOBFormat, value)

  /** @group setParam */
  def setIncludeEdges(value: Boolean): this.type = set(includeEdges, value)

  /** @group setParam */
  def setDelimiter(value: String): this.type = set(delimiter, value)

  /** @group setParam */
  def setPosModel(value: Array[String]): this.type = set(posModel, value)

  /** @group setParam */
  def setDependencyParserModel(value: Array[String]): this.type =
    set(dependencyParserModel, value)

  /** @group setParam */
  def setTypedDependencyParserModel(value: Array[String]): this.type =
    set(typedDependencyParserModel, value)

  setDefault(
    entityTypes -> Array(),
    explodeEntities -> false,
    maxSentenceSize -> 1000,
    minSentenceSize -> 2,
    mergeEntities -> false,
    rootTokens -> Array(),
    relationshipTypes -> Array(),
    includeEdges -> true,
    delimiter -> ",",
    posModel -> Array(),
    dependencyParserModel -> Array(),
    typedDependencyParserModel -> Array(),
    mergeEntitiesIOBFormat -> "IOB2")

  private lazy val allowedEntityRelationships = $(entityTypes).map { entityRelationship =>
    val result = entityRelationship.split("-")
    (result.head, result.last)
  }.distinct

  private lazy val allowedRelationshipTypes = $(relationshipTypes).map { relationshipTypes =>
    val result = relationshipTypes.split("-")
    (result.head, result.last)
  }.distinct

  private var pretrainedPos: Option[PerceptronModel] = None
  private var pretrainedDependencyParser: Option[DependencyParserModel] = None
  private var pretrainedTypedDependencyParser: Option[TypedDependencyParserModel] =
    None

  override def _transform(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): DataFrame = {
    if ($(mergeEntities)) {
      super._transform(dataset, recursivePipeline)
    } else {
      val structFields = dataset.schema.fields
        .filter(field => field.metadata.contains("annotatorType"))
        .filter(field =>
          field.metadata.getString("annotatorType") == DEPENDENCY ||
            field.metadata.getString("annotatorType") == LABELED_DEPENDENCY)
      if (structFields.length < 2) {
        throw new IllegalArgumentException(
          s"Missing either $DEPENDENCY or $LABELED_DEPENDENCY annotators. " +
            s"Make sure such annotators exist in your pipeline or setMergeEntities parameter to True")
      }

      val columnNames = structFields.map(structField => structField.name)
      val inputCols = getInputCols ++ columnNames
      val processedDataset = dataset.withColumn(
        getOutputCol,
        wrapColumnMetadata(dfAnnotate(array(inputCols.map(c => dataset.col(c)): _*))))
      processedDataset
    }
  }

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {

    if ($(mergeEntities)) {
      pretrainedPos = Some(PretrainedAnnotations.getPretrainedPos($(posModel)))
      pretrainedDependencyParser = Some(
        PretrainedAnnotations.getDependencyParser($(dependencyParserModel)))
      pretrainedTypedDependencyParser = Some(TypedDependencyParserModel.pretrained())
    }

    dataset
  }

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentenceIndexesToSkip = annotations
      .filter(_.annotatorType == AnnotatorType.DOCUMENT)
      .filter(annotation =>
        annotation.result.length > $(maxSentenceSize) || annotation.result.length < $(
          minSentenceSize))
      .map(annotation => annotation.metadata("sentence"))
      .toList
      .distinct

    val annotationsToProcess = annotations.filter(annotation => {
      !sentenceIndexesToSkip.contains(annotation.metadata.getOrElse("sentence", "0"))
    })

    if (annotationsToProcess.isEmpty) {
      Seq(Annotation(NODE, 0, 0, "", Map()))
    } else {
      computeAnnotatePaths(annotationsToProcess)
    }
  }

  private def computeAnnotatePaths(annotations: Seq[Annotation]): Seq[Annotation] = {
    val annotationsBySentence = annotations
      .groupBy(token => token.metadata.getOrElse("sentence", "0").toInt)
      .toSeq
      .sortBy(_._1)
      .map(annotationBySentence => annotationBySentence._2)

    val graphPaths = annotationsBySentence.flatMap { sentenceAnnotations =>
      val annotationsWithDependencies = getAnnotationsWithDependencies(sentenceAnnotations)
      val tokens = annotationsWithDependencies.filter(_.annotatorType == AnnotatorType.TOKEN)
      val nerEntities = annotationsWithDependencies.filter(annotation =>
        annotation.annotatorType == TOKEN && annotation.metadata("entity") != "O")

      if (nerEntities.isEmpty) {
        Seq(Annotation(NODE, 0, 0, "", Map()))
      } else {
        val dependencyData = LabeledDependency.unpackHeadAndRelation(annotationsWithDependencies)
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

  private def getAnnotationsWithDependencies(
      sentenceAnnotations: Seq[Annotation]): Seq[Annotation] = {
    if ($(mergeEntities)) {
      getPretrainedAnnotations(sentenceAnnotations)
    } else {
      getEntityAnnotations(sentenceAnnotations)
    }
  }

  private def getPretrainedAnnotations(annotationsToProcess: Seq[Annotation]): Seq[Annotation] = {

    val relatedAnnotatedTokens = mergeRelatedTokens(annotationsToProcess)
    val sentence = annotationsToProcess.filter(_.annotatorType == AnnotatorType.DOCUMENT)

    val posInput = sentence ++ relatedAnnotatedTokens
    val posAnnotations = PretrainedAnnotations.getPosOutput(posInput, pretrainedPos.get)

    val dependencyParserInput = sentence ++ relatedAnnotatedTokens ++ posAnnotations
    val dependencyParserAnnotations =
      PretrainedAnnotations.getDependencyParserOutput(
        dependencyParserInput,
        pretrainedDependencyParser.get)

    val typedDependencyParserInput =
      relatedAnnotatedTokens ++ posAnnotations ++ dependencyParserAnnotations
    val typedDependencyParserAnnotations = PretrainedAnnotations.getTypedDependencyParserOutput(
      typedDependencyParserInput,
      pretrainedTypedDependencyParser.get)

    relatedAnnotatedTokens ++ dependencyParserAnnotations ++ typedDependencyParserAnnotations
  }

  private def getEntityAnnotations(annotationsToProcess: Seq[Annotation]): Seq[Annotation] = {
    val entityAnnotations = annotationsToProcess.filter(_.annotatorType == NAMED_ENTITY)
    val tokensWithEntity =
      annotationsToProcess.filter(_.annotatorType == TOKEN).zipWithIndex.map {
        case (annotation, index) =>
          val tag = entityAnnotations(index).result
          val entity = if (tag.length == 1) tag else tag.substring(2)
          val metadata = annotation.metadata ++ Map("entity" -> entity)
          Annotation(
            annotation.annotatorType,
            annotation.begin,
            annotation.end,
            annotation.result,
            metadata)
      }
    val dependencyParserAnnotations = annotationsToProcess.filter(annotation =>
      annotation.annotatorType == DEPENDENCY || annotation.annotatorType == LABELED_DEPENDENCY)

    tokensWithEntity ++ dependencyParserAnnotations
  }

  private def mergeRelatedTokens(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = NerTagged.unpack(annotations)
    val docs = annotations.filter(a =>
      a.annotatorType == AnnotatorType.DOCUMENT && sentences.exists(b =>
        b.indexedTaggedWords.exists(c => c.begin >= a.begin && c.end <= a.end)))

    val entities = sentences.zip(docs.zipWithIndex).flatMap { case (sentence, doc) =>
      NerTagsEncoding.fromIOB(
        sentence,
        doc._1,
        sentenceIndex = doc._2,
        includeNoneEntities = true,
        format = $(mergeEntitiesIOBFormat))
    }

    entities.map(entity =>
      Annotation(
        TOKEN,
        entity.start,
        entity.end,
        entity.text,
        Map("sentence" -> entity.sentenceId, "entity" -> entity.entity)))
  }

  private def extractGraphsFromEntities(
      annotationsInfo: AnnotationsInfo,
      graph: GraphBuilder): Seq[Annotation] = {
    var rootIndices: Array[Int] = Array()

    if ($(rootTokens).isEmpty) {
      val sourceDependency =
        annotationsInfo.dependencyData.filter(dependencyInfo => dependencyInfo.headIndex == 0)
      rootIndices = Array(annotationsInfo.dependencyData.indexOf(sourceDependency.head) + 1)
    } else {
      val sourceDependencies = $(rootTokens).flatMap(rootToken =>
        annotationsInfo.dependencyData.filter(_.token == rootToken))
      rootIndices = sourceDependencies.map(sourceDependency =>
        annotationsInfo.dependencyData.indexOf(sourceDependency) + 1)
    }

    val entitiesPairData =
      getEntitiesData(annotationsInfo.nerEntities, annotationsInfo.dependencyData)
    val annotatedPaths = rootIndices.flatMap(rootIndex =>
      getAnnotatedPaths(entitiesPairData, graph, rootIndex, annotationsInfo))
    annotatedPaths
  }

  private def extractGraphsFromRelationships(
      annotationsInfo: AnnotationsInfo,
      graph: GraphBuilder): Seq[Annotation] = {

    val annotatedGraphPaths = allowedRelationshipTypes.flatMap { relationshipTypes =>
      val rootData = annotationsInfo.tokens
        .filter(_.result == relationshipTypes._1)
        .map(token => (token, annotationsInfo.tokens.indexOf(token) + 1))
      val entityIndexes = annotationsInfo.nerEntities
        .filter(_.metadata("entity") == relationshipTypes._2)
        .map(nerEntity => annotationsInfo.tokens.indexOf(nerEntity) + 1)

      rootData.flatMap { rootInfo =>
        val paths = entityIndexes.flatMap(entityIndex =>
          buildPath(graph, (rootInfo._2, entityIndex), annotationsInfo.dependencyData))
        val pathsMap = paths.zipWithIndex.flatMap { case (path, index) =>
          Map(s"path${(index + 1).toString}" -> path)
        }.toMap
        if (paths.nonEmpty) {
          Some(
            Annotation(
              NODE,
              rootInfo._1.begin,
              rootInfo._1.end,
              rootInfo._1.result,
              Map(
                "relationship" -> s"${rootInfo._1.result},${relationshipTypes._2}") ++ pathsMap))
        } else {
          None
        }
      }
    }
    annotatedGraphPaths
  }

  private def buildPath(
      graph: GraphBuilder,
      nodesIndexes: (Int, Int),
      dependencyData: Seq[DependencyInfo]): Option[String] = {
    val rootIndex = nodesIndexes._1
    val nodesIndexesPath = graph.findPath(rootIndex, nodesIndexes._2)
    val path = nodesIndexesPath.map { nodeIndex =>
      val dependencyInfo = dependencyData(nodeIndex - 1)
      val relation = dependencyInfo.relation
      var result = dependencyInfo.token
      if ($(includeEdges)) {
        val edge =
          if (relation == "*root*" || nodeIndex == rootIndex) "" else relation + $(delimiter)
        result = edge + dependencyInfo.token
      }
      result
    }
    if (path.isEmpty) None else Some(path.mkString($(delimiter)))
  }

  private def getAnnotatedPaths(
      entitiesPairData: List[EntitiesPairInfo],
      graph: GraphBuilder,
      rootIndex: Int,
      annotationsInfo: AnnotationsInfo): Seq[Annotation] = {

    val tokens = annotationsInfo.tokens
    val dependencyData = annotationsInfo.dependencyData

    val paths = entitiesPairData.flatMap { entitiesPairInfo =>
      val leftPath =
        buildPath(graph, (rootIndex, entitiesPairInfo.entitiesIndex._1), dependencyData)
      val rightPath =
        buildPath(graph, (rootIndex, entitiesPairInfo.entitiesIndex._2), dependencyData)
      if (leftPath.nonEmpty && rightPath.nonEmpty) {
        Some(GraphInfo(entitiesPairInfo.entities, leftPath, rightPath))
      } else None
    }

    val sourceToken = tokens(rootIndex - 1)
    val annotatedPaths = paths.map { path =>
      val leftEntity = path.entities._1
      val rightEntity = path.entities._2
      val leftPathTokens = path.leftPath
      val rightPathTokens = path.rightPath

      Annotation(
        NODE,
        sourceToken.begin,
        sourceToken.end,
        sourceToken.result,
        Map(
          "entities" -> s"$leftEntity,$rightEntity",
          "left_path" -> leftPathTokens.mkString($(delimiter)),
          "right_path" -> rightPathTokens.mkString($(delimiter))))
    }
    annotatedPaths
  }

  private def getEntitiesData(
      annotatedEntities: Seq[Annotation],
      dependencyData: Seq[DependencyInfo]): List[EntitiesPairInfo] = {
    var annotatedEntitiesPairs: List[(Annotation, Annotation)] = List()
    if (allowedEntityRelationships.isEmpty) {
      annotatedEntitiesPairs =
        annotatedEntities.combinations(2).map(entity => (entity.head, entity.last)).toList
    } else {
      annotatedEntitiesPairs = allowedEntityRelationships
        .flatMap(entities => getAnnotatedNerEntitiesPairs(entities, annotatedEntities))
        .filter(entities =>
          entities._1.begin != entities._2.begin && entities._1.end != entities._2.end)
        .toList
    }

    val entitiesPairData = annotatedEntitiesPairs.map { annotatedEntityPair =>
      val dependencyInfoLeft = dependencyData.filter(dependencyInfo =>
        dependencyInfo.beginToken == annotatedEntityPair._1.begin && dependencyInfo.endToken == annotatedEntityPair._1.end)
      val dependencyInfoRight = dependencyData.filter(dependencyInfo =>
        dependencyInfo.beginToken == annotatedEntityPair._2.begin && dependencyInfo.endToken == annotatedEntityPair._2.end)
      val indexLeft = dependencyData.indexOf(dependencyInfoLeft.head) + 1
      val indexRight = dependencyData.indexOf(dependencyInfoRight.head) + 1

      EntitiesPairInfo(
        (indexLeft, indexRight),
        (annotatedEntityPair._1.metadata("entity"), annotatedEntityPair._2.metadata("entity")))
    }
    entitiesPairData.distinct
  }

  private def getAnnotatedNerEntitiesPairs(
      entities: (String, String),
      annotatedEntities: Seq[Annotation]): List[(Annotation, Annotation)] = {

    val leftEntities = annotatedEntities.filter(annotatedEntity =>
      annotatedEntity.metadata("entity") == entities._1)
    val rightEntities = annotatedEntities.filter(annotatedEntity =>
      annotatedEntity.metadata("entity") == entities._2)

    if (leftEntities.length > rightEntities.length) {
      leftEntities.flatMap { leftEntity =>
        rightEntities.map(rightEntity => (leftEntity, rightEntity))
      }.toList
    } else {
      rightEntities.flatMap { rightEntity =>
        leftEntities.map(leftEntity => (leftEntity, rightEntity))
      }.toList
    }

  }

  private case class EntitiesPairInfo(entitiesIndex: (Int, Int), entities: (String, String))

  private case class GraphInfo(
      entities: (String, String),
      leftPath: Option[String],
      rightPath: Option[String])

  private case class AnnotationsInfo(
      tokens: Seq[Annotation],
      nerEntities: Seq[Annotation],
      dependencyData: Seq[DependencyInfo])

  /** Output annotator types: NODE
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = NODE

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  /** Input annotator types: DOCUMENT, TOKEN, NAMED_ENTITY
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, NAMED_ENTITY)

  override val optionalInputAnnotatorTypes: Array[String] = Array(DEPENDENCY, LABELED_DEPENDENCY)

}
