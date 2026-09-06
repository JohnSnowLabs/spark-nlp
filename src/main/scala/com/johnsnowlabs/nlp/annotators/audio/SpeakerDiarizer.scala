/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.ml.ai.{DiarizationOptions, SpeakerDiarization, Whisper}
import com.johnsnowlabs.ml.ai.util.Diarization.{ClusterState, SpeakerClustering}
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.notSupportedEngineError
import com.johnsnowlabs.ml.util.ONNX
import com.johnsnowlabs.nlp._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import java.io.{ByteArrayOutputStream, DataInputStream, DataOutputStream}
import java.util.concurrent.ConcurrentHashMap
import scala.collection.JavaConverters._

/** Diarizes audio: labels who is speaking, when, and — optionally — what they said.
  *
  * Bundles three ONNX sub-models behind a single annotator, the same "one class, multiple
  * internal graphs" shape as `VisionEncoderDecoderForImageCaptioning`: a speech/overlap
  * segmentation model, a speaker-embedding model, and (only when `transcribe` is `true`) a reused
  * Whisper encoder-decoder for ASR. Clustering, RTTM export, and confidence scoring around those
  * models are pure Scala — see `com.johnsnowlabs.ml.ai.SpeakerDiarization` and
  * `com.johnsnowlabs.ml.ai.util.Diarization.SpeakerClustering` for that logic.
  *
  * '''Input/output''': input is `AUDIO` (mono 16kHz float samples, the same contract every other
  * audio annotator in this package uses), output is `SPEAKER`: one `Annotation` per detected
  * speaker turn, `begin`/`end` in milliseconds, `metadata("speaker")` holding the speaker label
  * (an anonymous `SPEAKER_NN` unless matched against `setSpeakerGallery`),
  * `metadata("confidence")` a 0-1 margin-based confidence, and `result` holding that turn's
  * transcript when `transcribe` is `true` (the default) or an empty string when it is `false`.
  *
  * '''Stereo shortcut''': set `channelMode` to `"stereo"` and pass two input columns (one per
  * channel, via `setInputCols(Array(leftCol, rightCol))`) to skip segmentation/embedding/
  * clustering entirely and assign one speaker per channel — cheaper and more accurate than
  * rediscovering speaker identity the recording already encodes, for telephony-style audio.
  *
  * '''Streaming''': `streamingMode`/`sessionId` provide a convenience in-memory cluster-state
  * cache that is correct only on a single JVM (`local[*]`, repeated `LightPipeline.fullAnnotate`
  * calls, or a caller-forced `.coalesce(1)`) — `batchAnnotate` best-effort-detects and warns when
  * more than one executor is active. Every output `Annotation` also carries
  * `metadata("clusterStateSnapshot")`, a base64 blob a caller can thread back in via
  * `setStreamingPriorState` on the next call for the same session — the only mechanism that is
  * correct under arbitrary Spark scheduling. See the class-internal streaming notes in the design
  * this class implements for why both exist rather than just the cache.
  *
  * '''Speaker gallery / enrolled identity''': `setSpeakerGallery` takes enrolled name ->
  * reference embedding; matching cluster centroids are renamed to that name instead of an
  * anonymous ID. The gallery is deliberately '''not''' a Spark `Feature`/`Param` — every
  * registered `Feature` auto-persists on `.save()` (see `Feature.scala`), which would silently
  * write enrolled biometric voiceprints to disk on every save. `persistSpeakerGallery` (default
  * `false`) is the explicit opt-in required before `.save()` writes the gallery at all, and
  * `persistEmbeddings` (default `false`) is the equivalent opt-in for exposing each output turn's
  * own raw voice embedding on `Annotation.embeddings` — otherwise, that field is empty.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val diarizer = SpeakerDiarizer.pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("speakers")
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  */
class SpeakerDiarizer(override val uid: String)
    extends AnnotatorModel[SpeakerDiarizer]
    with HasBatchedAnnotateAudio[SpeakerDiarizer]
    with HasAudioFeatureProperties
    with WriteOnnxModel
    with HasEngine
    with HasGeneratorProperties
    with HasProtectedParams {

  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SPEAKER
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.AUDIO)

  private val logger = LoggerFactory.getLogger(this.getClass.getName)

  def this() = this(Identifiable.randomUID("SpeakerDiarizer"))

  /** Segmentation model sliding window size, in seconds (Default: `10.0`, the pyannote
    * segmentation-3.0 architecture's fixed window).
    * @group param
    */
  val windowDuration =
    new FloatParam(this, "windowDuration", "Segmentation model sliding window size, in seconds")

  /** @group setParam */
  def setWindowDuration(value: Float): this.type = set(windowDuration, value)

  /** @group getParam */
  def getWindowDuration: Float = $(windowDuration)

  /** Hop between sliding windows, in seconds (Default: `1.0`).
    * @group param
    */
  val stepDuration =
    new FloatParam(this, "stepDuration", "Hop between segmentation sliding windows, in seconds")

  /** @group setParam */
  def setStepDuration(value: Float): this.type = set(stepDuration, value)

  /** @group getParam */
  def getStepDuration: Float = $(stepDuration)

  /** Speech-activity onset probability cutoff (Default: `0.5`).
    * @group param
    */
  val onsetThreshold =
    new FloatParam(this, "onsetThreshold", "Speech-activity onset probability cutoff")

  /** @group setParam */
  def setOnsetThreshold(value: Float): this.type = set(onsetThreshold, value)

  /** @group getParam */
  def getOnsetThreshold: Float = $(onsetThreshold)

  /** Speech-activity offset probability cutoff (Default: `0.5`). Used for real onset/offset
    * hysteresis over the coverage-averaged, cross-window speech-activity curve (see
    * `com.johnsnowlabs.ml.ai.SpeakerDiarization.segmentAndEmbed`): once speech is detected via
    * `onsetThreshold`, it's considered ongoing until the activity curve drops below this value.
    * @group param
    */
  val offsetThreshold =
    new FloatParam(this, "offsetThreshold", "Speech-activity offset probability cutoff")

  /** @group setParam */
  def setOffsetThreshold(value: Float): this.type = set(offsetThreshold, value)

  /** @group getParam */
  def getOffsetThreshold: Float = $(offsetThreshold)

  /** Discard speech turns shorter than this, in seconds (Default: `0.0`).
    * @group param
    */
  val minDurationOn =
    new FloatParam(this, "minDurationOn", "Discard speech turns shorter than this (seconds)")

  /** @group setParam */
  def setMinDurationOn(value: Float): this.type = set(minDurationOn, value)

  /** @group getParam */
  def getMinDurationOn: Float = $(minDurationOn)

  /** Merge speech runs separated by a gap shorter than this, in seconds (Default: `0.0`).
    * @group param
    */
  val minDurationOff = new FloatParam(
    this,
    "minDurationOff",
    "Merge speech runs separated by a gap shorter than this (seconds)")

  /** @group setParam */
  def setMinDurationOff(value: Float): this.type = set(minDurationOff, value)

  /** @group getParam */
  def getMinDurationOff: Float = $(minDurationOff)

  /** Turns shorter than this, in seconds, still get a low-confidence fallback embedding rather
    * than being silently dropped (Default: `0.5`).
    * @group param
    */
  val minSegmentDuration = new FloatParam(
    this,
    "minSegmentDuration",
    "Turns shorter than this (seconds) still get a fallback embedding rather than being dropped")

  /** @group setParam */
  def setMinSegmentDuration(value: Float): this.type = set(minSegmentDuration, value)

  /** @group getParam */
  def getMinSegmentDuration: Float = $(minSegmentDuration)

  /** Which embedding backbone tier the loaded pretrained model was exported with (`"fast"` or
    * `"accurate"`) — set by `loadSavedModel`/`readModel` from the exported artifact, not
    * runtime-swappable: the tier is baked into which ONNX weights were loaded, the same way
    * Whisper's tiny/base/small/medium/large tiers are separate `pretrained(name = ...)` artifacts
    * rather than a settable param.
    * @group param
    */
  val embeddingModelSize: ProtectedParam[String] =
    new Param[String](this, "embeddingModelSize", "Embedding backbone tier of the loaded model")
      .setProtected()

  /** @group getParam */
  def getEmbeddingModelSize: String = $(embeddingModelSize)

  /** Exact speaker count, if known — skips threshold search entirely.
    * @group param
    */
  val numSpeakers = new IntParam(this, "numSpeakers", "Exact speaker count if known")

  /** @group setParam */
  def setNumSpeakers(value: Int): this.type = set(numSpeakers, value)

  /** @group getParam */
  def getNumSpeakers: Option[Int] = get(numSpeakers)

  /** Lower bound on cluster count when `numSpeakers` is unset (Default: `1`).
    * @group param
    */
  val minSpeakers = new IntParam(this, "minSpeakers", "Lower bound on cluster count")

  /** @group setParam */
  def setMinSpeakers(value: Int): this.type = set(minSpeakers, value)

  /** @group getParam */
  def getMinSpeakers: Int = $(minSpeakers)

  /** Upper bound on cluster count when `numSpeakers` is unset (Default: `20`).
    * @group param
    */
  val maxSpeakers = new IntParam(this, "maxSpeakers", "Upper bound on cluster count")

  /** @group setParam */
  def setMaxSpeakers(value: Int): this.type = set(maxSpeakers, value)

  /** @group getParam */
  def getMaxSpeakers: Int = $(maxSpeakers)

  /** Cosine-distance cutoff for agglomerative clustering, ignored if `numSpeakers` is set
    * (Default: `0.7`).
    * @group param
    */
  val clusteringThreshold =
    new FloatParam(this, "clusteringThreshold", "Cosine-distance cutoff for clustering")

  /** @group setParam */
  def setClusteringThreshold(value: Float): this.type = set(clusteringThreshold, value)

  /** @group getParam */
  def getClusteringThreshold: Float = $(clusteringThreshold)

  /** Max cosine distance for a cluster centroid to be renamed to a `setSpeakerGallery` entry
    * (Default: `0.25`). Used to be a hardcoded literal inside `SpeakerClustering.cluster` with no
    * way to reach it at all — tune this looser to make gallery matching more forgiving of
    * recording-condition drift, or tighter to reduce false-positive identity matches.
    * @group param
    */
  val galleryAcceptanceDistance = new FloatParam(
    this,
    "galleryAcceptanceDistance",
    "Max cosine distance for a cluster to be renamed to a speakerGallery entry")

  /** @group setParam */
  def setGalleryAcceptanceDistance(value: Float): this.type =
    set(galleryAcceptanceDistance, value)

  /** @group getParam */
  def getGalleryAcceptanceDistance: Float = $(galleryAcceptanceDistance)

  /** Explicit opt-in to serialize the enrolled speaker gallery on `.save()` (Default: `false`).
    * The gallery holds biometric voiceprints and is never persisted unless this is `true`.
    * @group param
    */
  val persistSpeakerGallery = new BooleanParam(
    this,
    "persistSpeakerGallery",
    "Explicit opt-in to serialize the speaker gallery on save")

  /** @group setParam */
  def setPersistSpeakerGallery(value: Boolean): this.type = set(persistSpeakerGallery, value)

  /** @group getParam */
  def getPersistSpeakerGallery: Boolean = $(persistSpeakerGallery)

  /** Explicit opt-in to populate each output `Annotation`'s `embeddings` field with that turn's
    * raw voice embedding (Default: `false`). A voice embedding is biometric data the same way an
    * enrolled `speakerGallery` entry is — this exists for the same reason `persistSpeakerGallery`
    * does, gating the same category of data rather than emitting it unconditionally.
    * @group param
    */
  val persistEmbeddings = new BooleanParam(
    this,
    "persistEmbeddings",
    "Explicit opt-in to populate each Annotation's embeddings field with its voice embedding")

  /** @group setParam */
  def setPersistEmbeddings(value: Boolean): this.type = set(persistEmbeddings, value)

  /** @group getParam */
  def getPersistEmbeddings: Boolean = $(persistEmbeddings)

  private var _speakerGallery: Map[String, Array[Float]] = Map.empty

  /** Enrolled name -> reference embedding. A cluster centroid within the acceptance distance of
    * an entry is renamed to that entry's key instead of an anonymous `SPEAKER_NN` label.
    * @group setParam
    */
  def setSpeakerGallery(gallery: Map[String, Array[Float]]): this.type = {
    _speakerGallery = gallery
    this
  }

  /** Java/Py4J-friendly overload: a Python `dict` of `str -> list[float]` crossing the Py4J
    * boundary arrives as a `java.util.Map[String, java.util.List[_]]` - a `dict` alone isn't
    * assignable to a Scala `Map` at all (calling the Scala overload above throws `Py4JException:
    * Method setSpeakerGallery(...) does not exist`), and Java generics erasure means the outer
    * map's declared element type is never actually enforced either: each nested Python
    * `list[float]` really arrives as a `java.util.ArrayList` of boxed numbers (`Double`, in
    * practice), not a real `Array[Float]` - converting only the outer map and trusting the
    * generic annotation compiles fine but throws `ClassCastException: java.util.ArrayList cannot
    * be cast to [F` the moment the value is actually used as `Array[Float]` (both failure modes
    * confirmed via real Python execution, not hypotheticals). Every element is converted
    * explicitly here via `Number`, not assumed to already be `Float`/`Double`/anything specific.
    * @group setParam
    */
  def setSpeakerGallery(gallery: java.util.Map[String, java.util.List[_]]): this.type = {
    val converted: Map[String, Array[Float]] = gallery.asScala.map { case (name, values) =>
      name -> values.asScala.map(_.asInstanceOf[Number].floatValue()).toArray
    }.toMap
    setSpeakerGallery(converted)
  }

  /** @group getParam */
  def getSpeakerGallery: Map[String, Array[Float]] = _speakerGallery

  /** Java/Py4J-friendly accessor: Py4J's default return-value conversion only recognizes standard
    * Java collection types, not Scala's own `Map` - a caller reading the plain
    * `getSpeakerGallery` back from Python gets an unusable opaque `JavaObject` reference instead
    * of a `dict`. This is the method the Python wrapper actually calls.
    * @group getParam
    */
  def getSpeakerGalleryJava: java.util.Map[String, Array[Float]] = _speakerGallery.asJava

  /** Removes one enrolled speaker from the gallery (e.g. to honor a deletion request). */
  def removeSpeakerFromGallery(name: String): this.type = {
    _speakerGallery = _speakerGallery - name
    this
  }

  /** Clears every enrolled speaker from the gallery. */
  def clearSpeakerGallery(): this.type = {
    _speakerGallery = Map.empty
    this
  }

  /** Whether to run ASR and return speaker-tagged transcript segments, or speaker turns only with
    * no transcript and no ASR model loaded (Default: `true`).
    * @group param
    */
  val transcribe =
    new BooleanParam(this, "transcribe", "Whether to run ASR and return transcript segments")

  /** @group setParam */
  def setTranscribe(value: Boolean): this.type = set(transcribe, value)

  /** @group getParam */
  def getTranscribe: Boolean = $(transcribe)

  /** Optional target language for transcription, formatted like Whisper's own tokens (e.g.
    * `<|en|>`) — only meaningful when the bundled ASR model is multilingual. Unset lets the model
    * auto-detect.
    * @group param
    */
  val asrLanguage = new Param[String](
    this,
    "asrLanguage",
    "Optional target language for transcription, formatted like Whisper's language tokens (e.g. <|en|>)")

  /** @group setParam */
  def setAsrLanguage(value: String): this.type = {
    require(
      value.length == 6 && value.startsWith("<|") && value.endsWith("|>"),
      "asrLanguage must be a two-letter code enclosed like <|en|>")
    set(asrLanguage, value)
  }

  /** @group getParam */
  def getAsrLanguage: Option[String] = get(asrLanguage)

  /** Optional task for the bundled ASR model: `<|transcribe|>` (default behavior) or
    * `<|translate|>` (translate to English). Only meaningful for a multilingual Whisper export.
    * @group param
    */
  val asrTask = new Param[String](this, "asrTask", "<|transcribe|> or <|translate|>")

  /** @group setParam */
  def setAsrTask(value: String): this.type = {
    require(
      value == "<|translate|>" || value == "<|transcribe|>",
      "asrTask must be either '<|translate|>' or '<|transcribe|>'")
    set(asrTask, value)
  }

  /** @group getParam */
  def getAsrTask: Option[String] = get(asrTask)

  /** Enables the in-memory `sessionId`-keyed cluster-state cache (Default: `false`). Correct only
    * on a single JVM — see class scaladoc.
    * @group param
    */
  val streamingMode =
    new BooleanParam(this, "streamingMode", "Enable the in-memory streaming cluster-state cache")

  /** @group setParam */
  def setStreamingMode(value: Boolean): this.type = {
    set(streamingMode, value)
    if (value) warnIfUnsafeStreaming(SparkSession.getActiveSession)
    this
  }

  /** @group getParam */
  def getStreamingMode: Boolean = $(streamingMode)

  /** Session key for the `streamingMode` cache.
    * @group param
    */
  val sessionId = new Param[String](this, "sessionId", "Session key for the streaming cache")

  /** @group setParam */
  def setSessionId(value: String): this.type = set(sessionId, value)

  /** @group getParam */
  def getSessionId: Option[String] = get(sessionId)

  /** Audio context padded onto each side of an internal `maxChunkDurationSeconds` chunk, in
    * seconds (Default: `2.0`), so a speaker turn straddling a chunk boundary is captured whole by
    * at least one neighboring chunk instead of being cut into two turns — see
    * `com.johnsnowlabs.ml.ai.SpeakerDiarization.collectStitchedTurns`. Also used as the maximum
    * gap allowed between a chunk's leading turn and the previous chunk's trailing turn for them
    * to be considered the same, boundary-split turn.
    * @group param
    */
  val streamingContextSeconds = new FloatParam(
    this,
    "streamingContextSeconds",
    "Audio context padded onto each side of an internal chunk boundary, in seconds")

  /** @group setParam */
  def setStreamingContextSeconds(value: Float): this.type = set(streamingContextSeconds, value)

  /** @group getParam */
  def getStreamingContextSeconds: Float = $(streamingContextSeconds)

  /** Explicitly threads a serialized cluster-state blob (from a prior call's
    * `metadata("clusterStateSnapshot")`) into the next call for the same session — the
    * multi-executor-safe alternative to `streamingMode`'s in-memory cache.
    * @group setParam
    */
  def setStreamingPriorState(blob: String): this.type = {
    _explicitPriorState = Some(SpeakerClustering.deserializeState(blob))
    this
  }
  private var _explicitPriorState: Option[ClusterState] = None

  /** Mean overlap-class probability, over a turn's duration, above which it is flagged
    * `metadata("overlap") = "true"` (Default: `0.3`). Used to be a bare literal inside
    * `segmentAndEmbed` with no way to reach it.
    * @group param
    */
  val overlapThreshold =
    new FloatParam(this, "overlapThreshold", "Mean overlap-class probability that flags a turn")

  /** @group setParam */
  def setOverlapThreshold(value: Float): this.type = set(overlapThreshold, value)

  /** @group getParam */
  def getOverlapThreshold: Float = $(overlapThreshold)

  /** `"mono"` (default: run the full ML pipeline) or `"stereo"` (two input columns, one speaker
    * per channel, no ML at all).
    * @group param
    */
  val channelMode = new Param[String](this, "channelMode", "\"mono\" or \"stereo\"")

  /** @group setParam */
  def setChannelMode(value: String): this.type = {
    require(value == "mono" || value == "stereo", "channelMode must be \"mono\" or \"stereo\"")
    set(channelMode, value)
  }

  /** @group getParam */
  def getChannelMode: String = $(channelMode)

  /** Long audio is processed in chunks of this size, in seconds, with cluster centroids carried
    * across chunks (Default: `300.0`).
    * @group param
    */
  val maxChunkDurationSeconds = new FloatParam(
    this,
    "maxChunkDurationSeconds",
    "Long audio is chunked at this size, in seconds")

  /** @group setParam */
  def setMaxChunkDurationSeconds(value: Float): this.type = set(maxChunkDurationSeconds, value)

  /** @group getParam */
  def getMaxChunkDurationSeconds: Float = $(maxChunkDurationSeconds)

  /** A turn longer than this, in seconds, is cropped (from its start) before being fed to the
    * embedding model (Default: `30.0`). The bundled WeSpeaker embedding model was validated on
    * utterance-length audio, not arbitrarily long crops (e.g. one speaker with no detected pause
    * for minutes).
    * @group param
    */
  val maxEmbeddingClipSeconds = new FloatParam(
    this,
    "maxEmbeddingClipSeconds",
    "A turn longer than this (seconds) is cropped before embedding")

  /** @group setParam */
  def setMaxEmbeddingClipSeconds(value: Float): this.type = set(maxEmbeddingClipSeconds, value)

  /** @group getParam */
  def getMaxEmbeddingClipSeconds: Float = $(maxEmbeddingClipSeconds)

  /** A turn longer than this, in seconds, is cropped (from its start) before being transcribed
    * (Default: `30.0`). Whisper's encoder takes a fixed-size, exactly-30-second input window by
    * construction regardless of how much audio it's handed — this crops explicitly, with a logged
    * warning, rather than relying on whatever silent truncation the model does internally.
    * @group param
    */
  val maxAsrClipSeconds = new FloatParam(
    this,
    "maxAsrClipSeconds",
    "A turn longer than this (seconds) is cropped before transcription")

  /** @group setParam */
  def setMaxAsrClipSeconds(value: Float): this.type = set(maxAsrClipSeconds, value)

  /** @group getParam */
  def getMaxAsrClipSeconds: Float = $(maxAsrClipSeconds)

  setDefault(
    windowDuration -> 10.0f,
    stepDuration -> 1.0f,
    onsetThreshold -> 0.5f,
    offsetThreshold -> 0.5f,
    minDurationOn -> 0.0f,
    minDurationOff -> 0.0f,
    minSegmentDuration -> 0.5f,
    minSpeakers -> 1,
    maxSpeakers -> 20,
    clusteringThreshold -> 0.7f,
    galleryAcceptanceDistance -> 0.25f,
    persistSpeakerGallery -> false,
    persistEmbeddings -> false,
    transcribe -> true,
    streamingMode -> false,
    streamingContextSeconds -> 2.0f,
    overlapThreshold -> 0.3f,
    channelMode -> "mono",
    maxChunkDurationSeconds -> 300.0f,
    maxEmbeddingClipSeconds -> 30.0f,
    maxAsrClipSeconds -> 30.0f,
    batchSize -> 1,
    minOutputLength -> 0,
    maxOutputLength -> 448,
    doSample -> false,
    temperature -> 1.0,
    topK -> 1,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    beamSize -> 1,
    nReturnSequences -> 1)

  /** Applies a bundle of sensible defaults for a common recording shape, then returns `this` so
    * individual `set*` calls after it can still override any of them.
    *
    * @param profile
    *   one of `"call_center"` (2-party, stereo-first), `"meeting"` (several participants, mono),
    *   or `"podcast"` (few hosts, mono, favors precision over recall on speaker count)
    * @group setParam
    */
  def useProfile(profile: String): this.type = {
    profile match {
      case "call_center" =>
        setChannelMode("stereo").setTranscribe(true).setMinSpeakers(1).setMaxSpeakers(2)
      case "meeting" =>
        setChannelMode("mono")
          .setTranscribe(true)
          .setMinSpeakers(2)
          .setMaxSpeakers(12)
          .setClusteringThreshold(0.65f)
      case "podcast" =>
        setChannelMode("mono")
          .setTranscribe(true)
          .setMinSpeakers(1)
          .setMaxSpeakers(6)
          .setClusteringThreshold(0.7f)
      case other =>
        throw new IllegalArgumentException(
          s"Unknown profile '$other'. Expected one of: call_center, meeting, podcast")
    }
    this
  }

  private var _model: Option[Broadcast[SpeakerDiarization]] = None

  /** @group getParam */
  def getModelIfNotSet: SpeakerDiarization = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      segmentation: OnnxWrapper,
      embedding: OnnxWrapper,
      whisper: Option[Whisper]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new SpeakerDiarization(segmentation, embedding, whisper, getSamplingRate)))
    }
    this
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val model = getModelIfNotSet
    writeOnnxModels(
      path,
      spark,
      Seq(
        (model.segmentationWrapper, "segmentation_model"),
        (model.embeddingWrapper, "embedding_model")),
      SpeakerDiarizer.suffix)

    model.whisper.foreach { whisperModel =>
      val wrappers = whisperModel.onnxWrappers.getOrElse(
        throw new IllegalStateException(
          "SpeakerDiarizer's bundled ASR sub-model must be ONNX-backed"))
      writeOnnxModels(
        path,
        spark,
        Seq(
          (wrappers.encoder, "asr_encoder_model"),
          (wrappers.decoder, "asr_decoder_model"),
          (wrappers.decoderWithPast, "asr_decoder_with_past_model")),
        SpeakerDiarizer.suffix)
      AsrSubModelIO.write(path, spark, whisperModel)
    }

    if (getPersistSpeakerGallery) {
      SpeakerGalleryIO.write(path, spark, _speakerGallery)
    }
  }

  /** Best-effort, non-authoritative check: warns rather than silently risking inconsistent
    * speaker IDs when `streamingMode`'s in-memory cache is used somewhere it cannot be correct.
    * See the class scaladoc's streaming section. Called from `setStreamingMode`, not
    * `batchAnnotate`: this needs a real, populated `SparkSession`, which only the driver (where
    * every setter call happens) reliably has - `batchAnnotate` runs per-partition on executors,
    * where `SparkSession.getActiveSession` is always empty regardless of cluster size.
    */
  private def warnIfUnsafeStreaming(spark: Option[SparkSession]): Unit = {
    if (getStreamingMode) {
      val executorCount = spark.map(_.sparkContext.getExecutorMemoryStatus.size).getOrElse(1)
      if (executorCount > 1) {
        logger.warn(
          "streamingMode=true is set on a Spark session with more than one executor. The " +
            "in-memory sessionId cache is only correct on a single JVM (local[*], repeated " +
            "LightPipeline.fullAnnotate calls, or a caller-forced .coalesce(1)); speaker IDs may " +
            "be inconsistent across calls in this configuration. Use setStreamingPriorState with " +
            "the previous call's metadata(\"clusterStateSnapshot\") instead for a " +
            "multi-executor-safe alternative.")
      }
    }
  }

  override def batchAnnotate(
      batchedAnnotations: Seq[Array[AnnotationAudio]]): Seq[Seq[Annotation]] = {
    val hadExplicitPriorState = _explicitPriorState.isDefined
    val initialPriorState =
      _explicitPriorState.getOrElse(
        if (getStreamingMode)
          getSessionId
            .flatMap(id => Option(SpeakerDiarizer.streamingStateCache.get(id)))
            .getOrElse(ClusterState.empty)
        else ClusterState.empty)
    _explicitPriorState = None
    val isStreamingCall = getStreamingMode || hadExplicitPriorState

    val options = DiarizationOptions(
      windowDurationSeconds = getWindowDuration,
      stepDurationSeconds = getStepDuration,
      onsetThreshold = getOnsetThreshold,
      offsetThreshold = getOffsetThreshold,
      minDurationOnSeconds = getMinDurationOn,
      minDurationOffSeconds = getMinDurationOff,
      minSegmentDurationSeconds = getMinSegmentDuration,
      numSpeakers = getNumSpeakers,
      minSpeakers = getMinSpeakers,
      maxSpeakers = getMaxSpeakers,
      clusteringThreshold = getClusteringThreshold,
      maxChunkDurationSeconds = getMaxChunkDurationSeconds,
      channelMode = getChannelMode,
      transcribe = getTranscribe,
      asrMaxOutputLength = getMaxOutputLength,
      asrMinOutputLength = getMinOutputLength,
      asrDoSample = getDoSample,
      asrBeamSize = getBeamSize,
      asrNumReturnSequences = getNReturnSequences,
      asrTemperature = getTemperature,
      asrTopK = getTopK,
      asrTopP = getTopP,
      asrRepetitionPenalty = getRepetitionPenalty,
      asrNoRepeatNgramSize = getNoRepeatNgramSize,
      asrRandomSeed = getRandomSeed,
      asrTask = getAsrTask,
      asrLanguage = getAsrLanguage,
      speakerGallery = getSpeakerGallery,
      galleryAcceptanceDistance = getGalleryAcceptanceDistance,
      priorState = initialPriorState,
      overlapDetectionThreshold = getOverlapThreshold,
      chunkOverlapSeconds = getStreamingContextSeconds,
      persistEmbeddings = getPersistEmbeddings,
      forceExactSpeakerCount = !isStreamingCall,
      maxEmbeddingClipSeconds = getMaxEmbeddingClipSeconds,
      maxAsrClipSeconds = getMaxAsrClipSeconds)

    var runningState = initialPriorState
    batchedAnnotations.map { audioAnnotations =>
      if (audioAnnotations.nonEmpty) {
        val (annotationsPerInput, newState) =
          getModelIfNotSet.diarize(
            audioAnnotations.toSeq,
            options.copy(priorState = runningState))
        runningState = newState
        if (getStreamingMode) {
          getSessionId.foreach(id => SpeakerDiarizer.streamingStateCache.put(id, newState))
        }
        val snapshot = SpeakerClustering.serializeState(newState)
        annotationsPerInput.flatten.map { annotation =>
          annotation.copy(metadata = annotation.metadata + ("clusterStateSnapshot" -> snapshot))
        }
      } else Seq.empty
    }
  }
}

/** Writes/reads the enrolled speaker gallery as a small standalone file under the model's save
  * path, kept entirely separate from Spark ML's `Feature` auto-persistence mechanism (see
  * `SpeakerDiarizer`'s class scaladoc for why) so it is written if and only if
  * `persistSpeakerGallery` was explicitly set.
  */
private[audio] object SpeakerGalleryIO {
  private val fileName = "speaker_gallery.bin"

  def write(path: String, spark: SparkSession, gallery: Map[String, Array[Float]]): Unit = {
    val bytes = new ByteArrayOutputStream()
    val out = new DataOutputStream(bytes)
    out.writeInt(gallery.size)
    gallery.foreach { case (name, embedding) =>
      val nameBytes = name.getBytes("UTF-8")
      out.writeInt(nameBytes.length)
      out.write(nameBytes)
      out.writeInt(embedding.length)
      embedding.foreach(out.writeFloat)
    }
    out.flush()

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val out2 = fs.create(new Path(path, fileName), true)
    try {
      out2.write(bytes.toByteArray)
    } finally {
      out2.close()
    }
  }

  def read(path: String, spark: SparkSession): Map[String, Array[Float]] = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val galleryPath = new Path(path, fileName)
    if (!fs.exists(galleryPath)) Map.empty
    else {
      val in = fs.open(galleryPath)
      try {
        val dataIn = new DataInputStream(in)
        val count = dataIn.readInt()
        (0 until count).map { _ =>
          val nameLen = dataIn.readInt()
          val nameBytes = new Array[Byte](nameLen)
          dataIn.readFully(nameBytes)
          val name = new String(nameBytes, "UTF-8")
          val embLen = dataIn.readInt()
          val embedding = Array.fill(embLen)(dataIn.readFloat())
          name -> embedding
        }.toMap
      } finally {
        in.close()
      }
    }
  }
}

/** The non-ONNX-weight configuration the bundled ASR sub-model needs to reconstruct a working
  * `Whisper` instance on read — vocabulary, generation-time forced/suppressed tokens, and the
  * mel-spectrogram preprocessor config. `Whisper`'s own constructor only exposes these as public
  * `val`s (not `Feature`s), so `SpeakerDiarizer` — unlike `WhisperForCTC`, which keeps its own
  * `Feature` copies — persists them directly here rather than duplicating that state as
  * additional `Feature`s on this class.
  */
private[audio] case class AsrConfig(
    preprocessor: com.johnsnowlabs.nlp.annotators.audio.feature_extractor.WhisperPreprocessor,
    vocabulary: Map[String, Int],
    addedSpecialTokens: Map[String, Int],
    generationConfig: com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig)

private[audio] object AsrSubModelIO {
  import org.json4s._
  import org.json4s.jackson.JsonMethods._
  import org.json4s.jackson.Serialization
  implicit val formats: DefaultFormats.type = DefaultFormats

  private val fileName = "asr_config.json"

  /** Whether an ASR sub-model bundle was saved at `path` at all - checked up front so `readModel`
    * can tell "no bundle was ever saved here" (expected, silent `None`) apart from "a bundle was
    * saved here but failed to load" (a real bug: corrupt/malformed data, an incompatible export,
    * etc.) which should surface as a thrown exception instead of being silently downgraded to the
    * same "no ASR" outcome.
    */
  def exists(path: String, spark: SparkSession): Boolean = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    fs.exists(new Path(path, fileName))
  }

  private case class Json(
      featureSize: Int,
      hopLength: Int,
      nFft: Int,
      nSamples: Int,
      paddingSide: String,
      paddingValue: Float,
      samplingRate: Int,
      vocabulary: Map[String, Int],
      addedSpecialTokens: Map[String, Int],
      bosId: Int,
      padId: Int,
      eosId: Int,
      vocabSize: Int,
      beginSuppressTokens: Option[Array[Int]],
      suppressTokenIds: Option[Array[Int]],
      forcedDecoderIds: Option[Array[Array[Int]]])

  def write(path: String, spark: SparkSession, whisperModel: Whisper): Unit = {
    val p = whisperModel.preprocessor
    val gc = whisperModel.generationConfig
    val json = Json(
      featureSize = p.feature_size,
      hopLength = p.hop_length,
      nFft = p.n_fft,
      nSamples = p.n_samples,
      paddingSide = p.padding_side,
      paddingValue = p.padding_value,
      samplingRate = p.sampling_rate,
      vocabulary = whisperModel.vocabulary,
      addedSpecialTokens = whisperModel.addedSpecialTokens,
      bosId = gc.bosId,
      padId = gc.padId,
      eosId = gc.eosId,
      vocabSize = gc.vocabSize,
      beginSuppressTokens = gc.beginSuppressTokens,
      suppressTokenIds = gc.suppressTokenIds,
      forcedDecoderIds = gc.forcedDecoderIds.map(_.map { case (a, b) => Array(a, b) }))

    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val out = fs.create(new Path(path, fileName), true)
    try {
      out.write(Serialization.write(json).getBytes("UTF-8"))
    } finally {
      out.close()
    }
  }

  def read(path: String, spark: SparkSession): AsrConfig = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val in = fs.open(new Path(path, fileName))
    val jsonString =
      try {
        scala.io.Source.fromInputStream(in, "UTF-8").mkString
      } finally {
        in.close()
      }
    val json = parse(jsonString).extract[Json]

    AsrConfig(
      preprocessor =
        new com.johnsnowlabs.nlp.annotators.audio.feature_extractor.WhisperPreprocessor(
          feature_size = json.featureSize,
          hop_length = json.hopLength,
          n_fft = json.nFft,
          n_samples = json.nSamples,
          padding_side = json.paddingSide,
          padding_value = json.paddingValue,
          sampling_rate = json.samplingRate),
      vocabulary = json.vocabulary,
      addedSpecialTokens = json.addedSpecialTokens,
      generationConfig = com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig(
        bosId = json.bosId,
        padId = json.padId,
        eosId = json.eosId,
        vocabSize = json.vocabSize,
        beginSuppressTokens = json.beginSuppressTokens,
        suppressTokenIds = json.suppressTokenIds,
        forcedDecoderIds = json.forcedDecoderIds.map(_.map(pair => (pair(0), pair(1))))))
  }
}

trait ReadablePretrainedSpeakerDiarizerModel
    extends ParamsAndFeaturesReadable[SpeakerDiarizer]
    with HasPretrained[SpeakerDiarizer] {
  override val defaultModelName: Some[String] = Some("speaker_diarizer_wespeaker")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): SpeakerDiarizer = super.pretrained()

  override def pretrained(name: String): SpeakerDiarizer = super.pretrained(name)

  override def pretrained(name: String, lang: String): SpeakerDiarizer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): SpeakerDiarizer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadSpeakerDiarizerDLModel extends ReadOnnxModel {
  this: ParamsAndFeaturesReadable[SpeakerDiarizer] =>

  override val onnxFile: String = "speaker_diarizer_onnx"
  val suffix: String = "_speaker_diarizer"

  def readModel(instance: SpeakerDiarizer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val coreWrappers =
          readOnnxModels(path, spark, Seq("segmentation_model", "embedding_model"), suffix)

        val whisperModel: Option[Whisper] =
          if (instance.getTranscribe && AsrSubModelIO.exists(path, spark)) {
            val asrWrappers = readOnnxModels(
              path,
              spark,
              Seq("asr_encoder_model", "asr_decoder_model", "asr_decoder_with_past_model"),
              suffix)
            val encoderDecoderWrappers = EncoderDecoderWrappers(
              asrWrappers("asr_encoder_model"),
              decoder = asrWrappers("asr_decoder_model"),
              decoderWithPast = asrWrappers("asr_decoder_with_past_model"))
            val asrConfig = AsrSubModelIO.read(path, spark)
            Some(
              new Whisper(
                None,
                Some(encoderDecoderWrappers),
                None,
                preprocessor = asrConfig.preprocessor,
                vocabulary = asrConfig.vocabulary,
                addedSpecialTokens = asrConfig.addedSpecialTokens,
                generationConfig = asrConfig.generationConfig))
          } else None

        instance.setModelIfNotSet(
          spark,
          coreWrappers("segmentation_model"),
          coreWrappers("embedding_model"),
          whisperModel)

        if (instance.getPersistSpeakerGallery) {
          instance.setSpeakerGallery(SpeakerGalleryIO.read(path, spark))
        }
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): SpeakerDiarizer =
    loadSavedModel(modelPath, spark, asrModelPath = None)

  /** Java/Py4J-friendly overload of the `Option[String]` variant below: a `null` `asrModelPath`
    * means no ASR bundling, matching how Python's `loadSavedModel(folder, spark,
    * asrModelPath=None)` calls through.
    */
  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      asrModelPath: String): SpeakerDiarizer =
    loadSavedModel(modelPath, spark, Option(asrModelPath))

  /** Loads segmentation + embedding ONNX weights from `modelPath`, optionally bundling ASR fusion
    * by pointing `asrModelPath` at a separate directory holding a standard `optimum-cli export
    * onnx`-shaped Whisper export (`encoder_model.onnx`, `decoder_model.onnx`,
    * `decoder_with_past_model.onnx`, `config.json`, `vocab.json`, `added_tokens.json`,
    * `preprocessor_config.json` — exactly what `WhisperForCTC.loadSavedModel` itself expects,
    * reused here rather than duplicated) so `setTranscribe(true)` has a real model to call.
    */
  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      asrModelPath: Option[String]): SpeakerDiarizer = {
    val annotatorModel = new SpeakerDiarizer()
    annotatorModel.set(annotatorModel.engine, ONNX.name)
    annotatorModel.set(annotatorModel.embeddingModelSize, "accurate")

    val segmentationWrapper = OnnxWrapper.read(
      spark,
      modelPath,
      zipped = false,
      useBundle = true,
      modelName = "segmentation_model",
      onnxFileSuffix = None)
    val embeddingWrapper = OnnxWrapper.read(
      spark,
      modelPath,
      zipped = false,
      useBundle = true,
      modelName = "embedding_model",
      onnxFileSuffix = None)

    val whisperModel = asrModelPath.map { path =>
      WhisperForCTC.loadSavedModel(path, spark).getModelIfNotSet
    }

    annotatorModel.setModelIfNotSet(spark, segmentationWrapper, embeddingWrapper, whisperModel)
    annotatorModel
  }
}

/** This is the companion object of [[SpeakerDiarizer]]. Please refer to that class for the
  * documentation.
  */
object SpeakerDiarizer
    extends ReadablePretrainedSpeakerDiarizerModel
    with ReadSpeakerDiarizerDLModel {

  /** Executor-local cluster-state cache for `streamingMode=true` — see the class scaladoc's
    * streaming section for why this is a convenience, not a distributed-correctness guarantee.
    */
  private[audio] val streamingStateCache = new ConcurrentHashMap[String, ClusterState]()
}
