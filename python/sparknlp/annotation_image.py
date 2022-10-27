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

"""Contains the AnnotationImage data format
"""


class AnnotationImage:
    """Represents the output of Spark NLP Annotators for image output and their details.

    Parameters
    ----------
    annotator_type : str
        The type of the output of the annotator. Possible values are ``IMAGE``.
    origin: str
        * Represents the source URI of the image
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    nChannels: int
        Number of color channels
    mode: int
        OpenCV type
    result : list(bytes)
        Image data in bytes
    metadata : dict
        Associated metadata for this annotation
    """

    def __init__(self, annotatorType, origin, height, width, nChannels, mode, result, metadata):
        self.annotatorType = annotatorType
        self.origin = origin
        self.height = height
        self.width = width
        self.nChannels = nChannels
        self.mode = mode
        self.result = result
        self.metadata = metadata

    def copy(self, result):
        """Creates new AnnotationImage with a different result, containing all
        settings of this Annotation.

        Parameters
        ----------
        result : list(bytes)
            The result of the annotation that should be copied.

        Returns
        -------
        AnnotationImage
            Newly created AnnotationImage
        """
        return AnnotationImage(self.annotatorType, self.origin, self.height, self.width,
                               self.nChannels, self.mode, result, self.metadata)

    def __str__(self):
        return "AnnotationImage(%s, %s, %i, %i, %i, %i, %s, %s)" % (
            self.annotatorType,
            self.origin,
            self.height,
            self.width,
            self.nChannels,
            self.mode,
            str(self.result),
            str(self.metadata)
        )

    def __repr__(self):
        return self.__str__()
