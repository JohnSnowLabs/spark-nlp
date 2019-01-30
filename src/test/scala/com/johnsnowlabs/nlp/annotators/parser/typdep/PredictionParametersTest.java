package com.johnsnowlabs.nlp.annotators.parser.typdep;


import gnu.trove.map.hash.TObjectIntHashMap;
import org.junit.Ignore;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThat;

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