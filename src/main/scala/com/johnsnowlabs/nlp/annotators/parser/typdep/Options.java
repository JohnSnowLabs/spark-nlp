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

import java.io.Serializable;

public class Options implements Serializable {

    private static final long serialVersionUID = 1L;

    String unimapFile = null;

    int numberOfPreTrainingIterations = 2;
    private int numberOfTrainingIterations;
    boolean initTensorWithPretrain = true;
    float regularization = 0.01f;
    float gammaLabel = 0;
    int rankFirstOrderTensor = 50;
    int rankSecondOrderTensor = 30;

    public int getNumberOfTrainingIterations() {
        return numberOfTrainingIterations;
    }

    public void setNumberOfTrainingIterations(int numberOfTrainingIterations) {
        this.numberOfTrainingIterations = numberOfTrainingIterations;
    }

    public Options() {

    }

    private Options(Options options) {
        this.numberOfPreTrainingIterations = options.numberOfPreTrainingIterations;
        this.numberOfTrainingIterations = options.numberOfTrainingIterations;
        this.initTensorWithPretrain = options.initTensorWithPretrain;
        this.regularization = options.regularization;
        this.gammaLabel = options.gammaLabel;
        this.rankFirstOrderTensor = options.rankFirstOrderTensor;
        this.rankSecondOrderTensor = options.rankSecondOrderTensor;
    }

    static Options newInstance(Options options){
        //Copy factory
        return new Options(options);
    }

}
