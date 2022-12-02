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
"""Contains utilities for annotators."""

from sparknlp.common.read_as import ReadAs
import sparknlp.internal as _internal


def ExternalResource(path, read_as=ReadAs.TEXT, options={}):
    """Returns a representation fo an External Resource.

    How the resource is read can be set with `read_as`.

    Parameters
    ----------
    path : str
        Path to the resource
    read_as : str, optional
        How to read the resource, by default ReadAs.TEXT
    options : dict, optional
        Options to read the resource, by default {}
    """
    return _internal._ExternalResource(path, read_as, options).apply()


def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()

