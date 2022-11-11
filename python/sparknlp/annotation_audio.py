#  Copyright 2017-2022 John Snow Labs
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

"""Contains the AnnotationAudio data format
"""


class AnnotationAudio:
    """Represents the output of Spark NLP Annotators for audio output and their details.

    Parameters
    ----------
    annotator_type : str
        The type of the output of the annotator. Possible values are ``AUDIO``.
    result : list(floats)
        Audio data in floats - already loaded/processed audio files
    metadata : dict
        Associated metadata for this annotation
    """

    def __init__(self, annotatorType, result, metadata):
        self.annotatorType = annotatorType
        self.result = result
        self.metadata = metadata

    def copy(self, result):
        """Creates new AnnotationAudio with a different result, containing all
        settings of this Annotation.

        Parameters
        ----------
        result : list(bytes)
            The result of the annotation that should be copied.

        Returns
        -------
        AnnotationAudio
            Newly created AnnotationAudio
        """
        return AnnotationAudio(self.annotatorType, result, self.metadata)

    def __str__(self):
        return "AnnotationAudio(%s, %s, %s)" % (
            self.annotatorType,
            str(self.result),
            str(self.metadata)
        )

    def __repr__(self):
        return self.__str__()
