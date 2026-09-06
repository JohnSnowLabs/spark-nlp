#  Copyright 2017-2024 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Contains classes concerning SpeakerDiarizer."""

from sparknlp.common import *


class SpeakerDiarizer(AnnotatorModel,
                       HasBatchedAnnotateAudio,
                       HasAudioFeatureProperties,
                       HasEngine,
                       HasGeneratorProperties):
    """Diarizes audio: labels who is speaking, when, and - optionally - what they said.

    Bundles three ONNX sub-models behind a single annotator: a speech/overlap segmentation
    model, a speaker-embedding model, and (only when ``transcribe`` is ``True``) a reused
    Whisper encoder-decoder for ASR. Clustering, RTTM export, and confidence scoring around
    those models are pure Scala.

    Input is ``AUDIO`` (mono 16kHz float samples, the same contract every other audio
    annotator in this package uses). Output is ``SPEAKER``: one annotation per detected
    speaker turn, ``begin``/``end`` in milliseconds, ``metadata["speaker"]`` holding the
    speaker label (an anonymous ``SPEAKER_NN`` unless matched against ``setSpeakerGallery``),
    ``metadata["confidence"]`` a 0-1 margin-based confidence, and ``result`` holding that
    turn's transcript when ``transcribe`` is ``True`` (the default) or an empty string when
    it is ``False``.

    Set ``channelMode`` to ``"stereo"`` and pass two input columns (one per channel) to skip
    segmentation/embedding/clustering entirely and assign one speaker per channel - cheaper
    and more accurate than rediscovering speaker identity the recording already encodes, for
    telephony-style audio.

    ``streamingMode``/``sessionId`` provide a convenience in-memory cluster-state cache that
    is correct only on a single JVM (``local[*]``, repeated ``LightPipeline.fullAnnotate``
    calls, or a caller-forced ``.coalesce(1)``). Every output annotation also carries
    ``metadata["clusterStateSnapshot"]``, a base64 blob that can be threaded back in via
    ``setStreamingPriorState`` on the next call for the same session - the mechanism that is
    correct under arbitrary Spark scheduling.

    ``setSpeakerGallery`` takes enrolled name -> reference embedding; matching cluster
    centroids are renamed to that name instead of an anonymous ID. The gallery holds
    biometric voiceprints and is never persisted on ``.save()`` unless
    ``persistSpeakerGallery`` is explicitly set to ``True``. Similarly, each turn's own raw
    voice embedding is left off ``result.embeddings`` unless ``persistEmbeddings`` is
    explicitly set to ``True`` - it too is biometric data.

    ``beamSize`` and ``nReturnSequences`` (inherited from ``HasGeneratorProperties``) are
    accepted but currently have no effect: the bundled Whisper model always uses greedy
    decoding. Separately, ``topK`` values of 100 or less all behave like exactly 100, a
    hardcoded floor in the shared top-k sampling component this is built on.

    Pretrained models can be loaded with ``pretrained`` of the companion object:

    .. code-block:: python

        diarizer = SpeakerDiarizer.pretrained() \\
            .setInputCols(["audio_assembler"]) \\
            .setOutputCol("speakers")

    The default model is ``"speaker_diarizer_wespeaker"``, if no name is provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``AUDIO``              ``SPEAKER``
    ====================== ======================

    Parameters
    ----------
    windowDuration
        Segmentation model sliding window size, in seconds
    stepDuration
        Hop between segmentation sliding windows, in seconds
    onsetThreshold
        Speech-activity onset probability cutoff
    offsetThreshold
        Speech-activity offset probability cutoff
    minDurationOn
        Discard speech turns shorter than this (seconds)
    minDurationOff
        Merge speech runs separated by a gap shorter than this (seconds)
    minSegmentDuration
        Turns shorter than this (seconds) still get a fallback embedding rather than being
        dropped
    numSpeakers
        Exact speaker count if known - skips threshold search
    minSpeakers
        Lower bound on cluster count when numSpeakers is unset
    maxSpeakers
        Upper bound on cluster count when numSpeakers is unset
    clusteringThreshold
        Cosine-distance cutoff for agglomerative clustering, ignored if numSpeakers is set
    galleryAcceptanceDistance
        Max cosine distance for a cluster to be renamed to a speakerGallery entry
    persistSpeakerGallery
        Explicit opt-in to serialize the enrolled speaker gallery on save
    persistEmbeddings
        Explicit opt-in to populate each Annotation's embeddings field with its voice embedding
    transcribe
        Whether to run ASR and return transcript segments, or speaker turns only
    asrLanguage
        Optional target language for transcription, formatted like Whisper's language tokens
    asrTask
        "<|transcribe|>" or "<|translate|>"
    streamingMode
        Enable the in-memory streaming cluster-state cache
    sessionId
        Session key for the streaming cache
    streamingContextSeconds
        Audio context padded onto each side of an internal chunk boundary, in seconds
    overlapThreshold
        Mean overlap-class probability that flags a turn
    channelMode
        "mono" (run the full ML pipeline) or "stereo" (one speaker per input column, no ML)
    maxChunkDurationSeconds
        Long audio is chunked at this size, in seconds
    maxEmbeddingClipSeconds
        A turn longer than this (seconds) is cropped before embedding
    maxAsrClipSeconds
        A turn longer than this (seconds) is cropped before transcription

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> audioAssembler = AudioAssembler() \\
    ...     .setInputCol("audio_content") \\
    ...     .setOutputCol("audio_assembler")
    >>> diarizer = SpeakerDiarizer.pretrained() \\
    ...     .setInputCols(["audio_assembler"]) \\
    ...     .setOutputCol("speakers")
    >>> pipeline = Pipeline().setStages([audioAssembler, diarizer])
    >>> processedAudioFloats = spark.createDataFrame([[rawFloats]]).toDF("audio_content")
    >>> result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
    >>> result.selectExpr("explode(speakers) as s").selectExpr(
    ...     "s.begin", "s.end", "s.metadata['speaker']", "s.result").show(truncate=False)
    """
    name = "SpeakerDiarizer"

    inputAnnotatorTypes = [AnnotatorType.AUDIO]

    outputAnnotatorType = AnnotatorType.SPEAKER

    windowDuration = Param(Params._dummy(), "windowDuration",
                            "Segmentation model sliding window size, in seconds",
                            typeConverter=TypeConverters.toFloat)

    stepDuration = Param(Params._dummy(), "stepDuration",
                          "Hop between segmentation sliding windows, in seconds",
                          typeConverter=TypeConverters.toFloat)

    onsetThreshold = Param(Params._dummy(), "onsetThreshold",
                            "Speech-activity onset probability cutoff",
                            typeConverter=TypeConverters.toFloat)

    offsetThreshold = Param(Params._dummy(), "offsetThreshold",
                             "Speech-activity offset probability cutoff",
                             typeConverter=TypeConverters.toFloat)

    minDurationOn = Param(Params._dummy(), "minDurationOn",
                           "Discard speech turns shorter than this (seconds)",
                           typeConverter=TypeConverters.toFloat)

    minDurationOff = Param(Params._dummy(), "minDurationOff",
                            "Merge speech runs separated by a gap shorter than this (seconds)",
                            typeConverter=TypeConverters.toFloat)

    minSegmentDuration = Param(Params._dummy(), "minSegmentDuration",
                                "Turns shorter than this (seconds) still get a fallback "
                                "embedding rather than being dropped",
                                typeConverter=TypeConverters.toFloat)

    numSpeakers = Param(Params._dummy(), "numSpeakers",
                         "Exact speaker count if known",
                         typeConverter=TypeConverters.toInt)

    minSpeakers = Param(Params._dummy(), "minSpeakers",
                         "Lower bound on cluster count",
                         typeConverter=TypeConverters.toInt)

    maxSpeakers = Param(Params._dummy(), "maxSpeakers",
                         "Upper bound on cluster count",
                         typeConverter=TypeConverters.toInt)

    clusteringThreshold = Param(Params._dummy(), "clusteringThreshold",
                                 "Cosine-distance cutoff for clustering",
                                 typeConverter=TypeConverters.toFloat)

    galleryAcceptanceDistance = Param(Params._dummy(), "galleryAcceptanceDistance",
                                       "Max cosine distance for a cluster to be renamed to a "
                                       "speakerGallery entry",
                                       typeConverter=TypeConverters.toFloat)

    persistSpeakerGallery = Param(Params._dummy(), "persistSpeakerGallery",
                                   "Explicit opt-in to serialize the speaker gallery on save",
                                   typeConverter=TypeConverters.toBoolean)

    persistEmbeddings = Param(Params._dummy(), "persistEmbeddings",
                               "Explicit opt-in to populate each Annotation's embeddings field "
                               "with its voice embedding",
                               typeConverter=TypeConverters.toBoolean)

    transcribe = Param(Params._dummy(), "transcribe",
                        "Whether to run ASR and return transcript segments",
                        typeConverter=TypeConverters.toBoolean)

    asrLanguage = Param(Params._dummy(), "asrLanguage",
                         "Optional target language for transcription, formatted like Whisper's "
                         "language tokens (e.g. <|en|>)",
                         typeConverter=TypeConverters.toString)

    asrTask = Param(Params._dummy(), "asrTask", "<|transcribe|> or <|translate|>",
                     typeConverter=TypeConverters.toString)

    streamingMode = Param(Params._dummy(), "streamingMode",
                           "Enable the in-memory streaming cluster-state cache",
                           typeConverter=TypeConverters.toBoolean)

    sessionId = Param(Params._dummy(), "sessionId",
                       "Session key for the streaming cache",
                       typeConverter=TypeConverters.toString)

    streamingContextSeconds = Param(Params._dummy(), "streamingContextSeconds",
                                     "Trailing-audio context carried across chunks, in seconds",
                                     typeConverter=TypeConverters.toFloat)

    overlapThreshold = Param(Params._dummy(), "overlapThreshold",
                              "Mean overlap-class probability that flags a turn",
                              typeConverter=TypeConverters.toFloat)

    channelMode = Param(Params._dummy(), "channelMode",
                         "\"mono\" or \"stereo\"",
                         typeConverter=TypeConverters.toString)

    maxChunkDurationSeconds = Param(Params._dummy(), "maxChunkDurationSeconds",
                                     "Long audio is chunked at this size, in seconds",
                                     typeConverter=TypeConverters.toFloat)

    maxEmbeddingClipSeconds = Param(Params._dummy(), "maxEmbeddingClipSeconds",
                                     "A turn longer than this (seconds) is cropped before "
                                     "embedding",
                                     typeConverter=TypeConverters.toFloat)

    maxAsrClipSeconds = Param(Params._dummy(), "maxAsrClipSeconds",
                               "A turn longer than this (seconds) is cropped before "
                               "transcription",
                               typeConverter=TypeConverters.toFloat)

    def setWindowDuration(self, value):
        """Sets the segmentation model sliding window size, in seconds."""
        return self._set(windowDuration=value)

    def setStepDuration(self, value):
        """Sets the hop between segmentation sliding windows, in seconds."""
        return self._set(stepDuration=value)

    def setOnsetThreshold(self, value):
        """Sets the speech-activity onset probability cutoff."""
        return self._set(onsetThreshold=value)

    def setOffsetThreshold(self, value):
        """Sets the speech-activity offset probability cutoff."""
        return self._set(offsetThreshold=value)

    def setMinDurationOn(self, value):
        """Sets the minimum speech turn duration, in seconds, below which turns are discarded."""
        return self._set(minDurationOn=value)

    def setMinDurationOff(self, value):
        """Sets the max gap, in seconds, below which adjacent speech runs are merged."""
        return self._set(minDurationOff=value)

    def setMinSegmentDuration(self, value):
        """Sets the minimum turn duration, in seconds, for a full-confidence embedding."""
        return self._set(minSegmentDuration=value)

    def setNumSpeakers(self, value):
        """Sets the exact speaker count, if known - skips threshold search."""
        return self._set(numSpeakers=value)

    def setMinSpeakers(self, value):
        """Sets the lower bound on cluster count."""
        return self._set(minSpeakers=value)

    def setMaxSpeakers(self, value):
        """Sets the upper bound on cluster count."""
        return self._set(maxSpeakers=value)

    def setClusteringThreshold(self, value):
        """Sets the cosine-distance cutoff for agglomerative clustering."""
        return self._set(clusteringThreshold=value)

    def setGalleryAcceptanceDistance(self, value):
        """Sets the max cosine distance for a cluster to be renamed to a speakerGallery entry."""
        return self._set(galleryAcceptanceDistance=value)

    def setPersistSpeakerGallery(self, value):
        """Sets whether to persist the enrolled speaker gallery on save."""
        return self._set(persistSpeakerGallery=value)

    def setPersistEmbeddings(self, value):
        """Sets whether to populate each output Annotation's embeddings field with its voice
        embedding. A voice embedding is biometric data, so this defaults to False."""
        return self._set(persistEmbeddings=value)

    def setTranscribe(self, value):
        """Sets whether to run ASR and return transcript segments."""
        return self._set(transcribe=value)

    def setAsrLanguage(self, value):
        """Sets the optional target language for transcription, formatted like Whisper's own
        tokens (e.g. "<|en|>") - only meaningful when the bundled ASR model is multilingual.

        Validated on the Python side (rather than relying on a round trip through
        _call_java(self.setAsrLanguage, ...)) so an invalid value fails fast with a normal Python
        exception, and so this class's own Python-side param tracking - which Params._set keeps in
        sync but an arbitrary custom Java method call would not - stays correct afterwards.
        """
        if not (len(value) == 6 and value.startswith("<|") and value.endswith("|>")):
            raise ValueError("asrLanguage must be a two-letter code enclosed like <|en|>")
        return self._set(asrLanguage=value)

    def getAsrLanguage(self):
        """Gets the optional target language for transcription, or None if it was never set.

        Mirrors the Scala side's own getAsrLanguage: Option[String], which returns None when
        unset rather than throwing. `getOrDefault("asrLanguage")` alone does not work for that -
        asrLanguage has no default value on either side (unset genuinely means "let the model
        auto-detect", not some fallback value), so plain getOrDefault raises a bare KeyError
        instead (confirmed via real execution, not a hypothetical) - there is no way for Python
        code to distinguish "not set" from "class is broken" without this method.
        """
        return self.getOrDefault("asrLanguage") if self.isSet(self.asrLanguage) else None

    def setAsrTask(self, value):
        """Sets the optional task for the bundled ASR model: "<|transcribe|>" (default behavior)
        or "<|translate|>" (translate to English). Only meaningful for a multilingual Whisper
        export. See setAsrLanguage's docstring for why this validates client-side."""
        if value not in ("<|translate|>", "<|transcribe|>"):
            raise ValueError("asrTask must be either '<|translate|>' or '<|transcribe|>'")
        return self._set(asrTask=value)

    def getAsrTask(self):
        """Gets the optional task for the bundled ASR model, or None if it was never set. See
        getAsrLanguage's docstring for why this exists instead of plain getOrDefault."""
        return self.getOrDefault("asrTask") if self.isSet(self.asrTask) else None

    def setStreamingMode(self, value):
        """Sets whether to enable the in-memory streaming cluster-state cache."""
        return self._set(streamingMode=value)

    def setSessionId(self, value):
        """Sets the session key for the streaming cache."""
        return self._set(sessionId=value)

    def setStreamingContextSeconds(self, value):
        """Sets the audio context padded onto each side of an internal chunk boundary, in
        seconds, so a turn straddling the boundary is stitched back into one instead of being
        cut into two."""
        return self._set(streamingContextSeconds=value)

    def setOverlapThreshold(self, value):
        """Sets the mean overlap-class probability, over a turn's duration, that flags it as
        metadata["overlap"] = "true"."""
        return self._set(overlapThreshold=value)

    def setChannelMode(self, value):
        """Sets "mono" (full ML pipeline) or "stereo" (one speaker per input column)."""
        return self._set(channelMode=value)

    def setMaxChunkDurationSeconds(self, value):
        """Sets the chunk size for long audio, in seconds."""
        return self._set(maxChunkDurationSeconds=value)

    def setMaxEmbeddingClipSeconds(self, value):
        """Sets the turn-length cap, in seconds, applied before embedding."""
        return self._set(maxEmbeddingClipSeconds=value)

    def setMaxAsrClipSeconds(self, value):
        """Sets the turn-length cap, in seconds, applied before transcription."""
        return self._set(maxAsrClipSeconds=value)

    def setSpeakerGallery(self, gallery):
        """Sets the enrolled speaker gallery: a dict of name -> list[float] reference embedding.

        Parameters
        ----------
        gallery : dict[str, list[float]]
            Enrolled name -> reference embedding. A cluster centroid within the acceptance
            distance of an entry is renamed to that entry's key.
        """
        normalized = {name: [float(v) for v in values] for name, values in gallery.items()}
        self._call_java("setSpeakerGallery", normalized)
        return self

    def getSpeakerGallery(self):
        """Gets the enrolled speaker gallery."""
        return self._call_java("getSpeakerGalleryJava")

    def removeSpeakerFromGallery(self, name):
        """Removes one enrolled speaker from the gallery."""
        self._call_java("removeSpeakerFromGallery", name)
        return self

    def clearSpeakerGallery(self):
        """Clears every enrolled speaker from the gallery."""
        self._call_java("clearSpeakerGallery")
        return self

    def setStreamingPriorState(self, blob):
        """Explicitly threads a serialized cluster-state blob into the next call for the same
        session - the multi-executor-safe alternative to streamingMode's in-memory cache.

        Parameters
        ----------
        blob : str
            A previous call's ``metadata["clusterStateSnapshot"]`` value
        """
        self._call_java("setStreamingPriorState", blob)
        return self

    def useProfile(self, profile):
        """Applies a bundle of sensible defaults for a common recording shape.

        Parameters
        ----------
        profile : str
            One of "call_center" (2-party, stereo-first), "meeting" (several participants,
            mono), or "podcast" (few hosts, mono)
        """
        if profile == "call_center":
            self.setChannelMode("stereo").setTranscribe(True).setMinSpeakers(1).setMaxSpeakers(2)
        elif profile == "meeting":
            self.setChannelMode("mono").setTranscribe(True).setMinSpeakers(2).setMaxSpeakers(
                12).setClusteringThreshold(0.65)
        elif profile == "podcast":
            self.setChannelMode("mono").setTranscribe(True).setMinSpeakers(1).setMaxSpeakers(
                6).setClusteringThreshold(0.7)
        else:
            raise ValueError(
                "Unknown profile '%s'. Expected one of: call_center, meeting, podcast" % profile)
        return self

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.audio.SpeakerDiarizer",
                 java_model=None):
        super(SpeakerDiarizer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            windowDuration=10.0,
            stepDuration=1.0,
            onsetThreshold=0.5,
            offsetThreshold=0.5,
            minDurationOn=0.0,
            minDurationOff=0.0,
            minSegmentDuration=0.5,
            minSpeakers=1,
            maxSpeakers=20,
            clusteringThreshold=0.7,
            galleryAcceptanceDistance=0.25,
            persistSpeakerGallery=False,
            persistEmbeddings=False,
            transcribe=True,
            streamingMode=False,
            streamingContextSeconds=2.0,
            overlapThreshold=0.3,
            channelMode="mono",
            maxChunkDurationSeconds=300.0,
            maxEmbeddingClipSeconds=30.0,
            maxAsrClipSeconds=30.0,
            batchSize=1,
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, asr_model_path=None):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model (segmentation + embedding ONNX weights)
        spark_session : pyspark.sql.SparkSession
            The current SparkSession
        asr_model_path : str, optional
            Path to a separate standard ``optimum-cli export onnx``-shaped Whisper export
            (same layout ``WhisperForCTC.loadSavedModel`` itself expects) to bundle for ASR
            fusion. Leave unset for diarization-only models; ``setTranscribe(True)`` requires
            this to have been provided.

        Returns
        -------
        SpeakerDiarizer
            The restored model
        """
        from sparknlp.internal import _SpeakerDiarizer
        jModel = _SpeakerDiarizer(folder, spark_session._jsparkSession, asr_model_path)._java_obj
        return SpeakerDiarizer(java_model=jModel)

    @staticmethod
    def pretrained(name="speaker_diarizer_wespeaker", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "speaker_diarizer_wespeaker"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SpeakerDiarizer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SpeakerDiarizer, name, lang, remote_loc)
