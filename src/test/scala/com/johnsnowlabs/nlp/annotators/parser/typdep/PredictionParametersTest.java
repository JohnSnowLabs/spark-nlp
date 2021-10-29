/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.parser.typdep;


import gnu.trove.map.hash.TObjectIntHashMap;
import org.junit.Test;

import static org.junit.Assert.assertFalse;

public class PredictionParametersTest {

    @Test
    public void shouldTransformToTroveMapWhenStringRepresentationWithTwoEqualsIsSent() {

        String mapAsString = "{form=<root>=1,form=The=2}";

        PredictionParameters predictionParameters = new PredictionParameters();
        TObjectIntHashMap map = predictionParameters.transformToTroveMap(mapAsString);

        assertFalse(map.isEmpty());

    }

    @Test
    public void shouldTransformToTroveMapWhenStringRepresentationWithOneEqualIsSent() {

        String mapAsString = "{TITLE=27,SBJ=4,PM0D=8}";

        PredictionParameters predictionParameters = new PredictionParameters();
        TObjectIntHashMap map = predictionParameters.transformToTroveMap(mapAsString);

        assertFalse(map.isEmpty());

    }

    @Test
    public void shouldTransformToTroveMapWhenStringRepresentationHasWrongFormat() {

        String mapAsString = "{}";

        PredictionParameters predictionParameters = new PredictionParameters();
        TObjectIntHashMap map = predictionParameters.transformToTroveMap(mapAsString);

        assertFalse(map.isEmpty());

    }

}