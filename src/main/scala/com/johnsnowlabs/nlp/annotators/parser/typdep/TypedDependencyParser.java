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

package com.johnsnowlabs.nlp.annotators.parser.typdep;

import com.johnsnowlabs.nlp.annotators.parser.typdep.io.ConllWriter;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.DependencyLabel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;


/**
 * Labeled parser that finds a grammatical relation between two words in a sentence. Its input is a CoNLL2009 or ConllU dataset.
 *
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
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
 * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 * @see [[com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach TypedDependencyParserApproach]]
 * @see [[com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel TypedDependencyParserModel]]
 */
public class TypedDependencyParser implements Serializable {

    private static final long serialVersionUID = 1L;
    private transient Logger logger = LoggerFactory.getLogger("TypedDependencyParser");
    /**
     * @group param
     **/
    private Options options;
    /**
     * @group param
     **/
    private DependencyPipe dependencyPipe;
    /**
     * @group param
     **/
    private Parameters parameters;

    /**
     * @group getParam
     **/
    DependencyPipe getDependencyPipe() {
        return dependencyPipe;
    }

    /**
     * @group getParam
     **/
    Parameters getParameters() {
        return parameters;
    }

    /**
     * @group getParam
     **/
    public Options getOptions() {
        return options;
    }

    /**
     * @group setParam
     **/
    void setDependencyPipe(DependencyPipe dependencyPipe) {
        this.dependencyPipe = dependencyPipe;
    }

    /**
     * @group setParam
     **/
    void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    /**
     * @group setParam
     **/
    public void setOptions(Options options) {
        this.options = options;
    }

    void train(DependencyInstance[] dependencyInstances) {
        long start;
        long end;

        if ((options.rankFirstOrderTensor > 0 || options.rankSecondOrderTensor > 0) && options.gammaLabel < 1
                && options.initTensorWithPretrain) {

            Options optionsBackup = Options.newInstance(options);
            options.rankFirstOrderTensor = 0;
            options.rankSecondOrderTensor = 0;
            options.gammaLabel = 1.0f;
            optionsBackup.setNumberOfTrainingIterations(options.numberOfPreTrainingIterations);
            parameters.setRankFirstOrderTensor(options.rankFirstOrderTensor);
            parameters.setRankSecondOrderTensor(options.rankSecondOrderTensor);
            parameters.setGammaLabel(options.gammaLabel);

            logger.debug("Pre-training:%n");

            start = System.currentTimeMillis();

            logger.debug("Running MIRA ... ");
            trainIterations(dependencyInstances);

            options = optionsBackup;
            parameters.setRankFirstOrderTensor(options.rankFirstOrderTensor);
            parameters.setRankSecondOrderTensor(options.rankSecondOrderTensor);
            parameters.setGammaLabel(options.gammaLabel);

            logger.debug("Init tensor ... ");
            int n = parameters.getNumberWordFeatures();
            int d = parameters.getDL();
            LowRankTensor tensor = new LowRankTensor(new int[]{n, n, d}, options.rankFirstOrderTensor);
            LowRankTensor tensor2 = new LowRankTensor(new int[]{n, n, n, d, d}, options.rankSecondOrderTensor);
            dependencyPipe.getSynFactory().fillParameters(tensor, tensor2, parameters);

            ArrayList<float[][]> param = new ArrayList<>();
            param.add(parameters.getU());
            param.add(parameters.getV());
            param.add(parameters.getWL());
            tensor.decompose(param);
            ArrayList<float[][]> param2 = new ArrayList<>();
            param2.add(parameters.getU2());
            param2.add(parameters.getV2());
            param2.add(parameters.getW2());
            param2.add(parameters.getX2L());
            param2.add(parameters.getY2L());
            tensor2.decompose(param2);
            parameters.assignTotal();
            parameters.printStat();

            end = System.currentTimeMillis();
            if (logger.isDebugEnabled()) {
                logger.debug(String.format("Pre-training took %d ms.%n", end - start));
            }

        } else {
            parameters.randomlyInit();
        }

        logger.debug(" Training:%n");

        start = System.currentTimeMillis();

        //TODO: Check if we really require to train here again
        logger.debug("Running MIRA ... ");
        trainIterations(dependencyInstances);

        end = System.currentTimeMillis();

        if (logger.isDebugEnabled()) {
            logger.debug(String.format("Training took %d ms.%n", end - start));
        }
    }

    private void trainIterations(DependencyInstance[] dependencyInstances) {
        int printPeriod = 10000 < dependencyInstances.length ? dependencyInstances.length / 10 : 1000;

        if (logger.isDebugEnabled()) {
            logger.debug(String.format("Number of Training Iterations: %d",
                    options.getNumberOfTrainingIterations()));
        }

        for (int iIter = 0; iIter < options.getNumberOfTrainingIterations(); ++iIter) {

            long start;
            double loss = 0;
            int totalNUmberCorrectMatches = 0;
            int tot = 0;
            start = System.currentTimeMillis();

            for (int i = 0; i < dependencyInstances.length; ++i) {

                if ((i + 1) % printPeriod == 0 && logger.isDebugEnabled()) {
                    logger.debug(String.format("  %d (time=%ds)", (i + 1),
                            (System.currentTimeMillis() - start) / 1000));
                }

                DependencyInstance dependencyInstance = dependencyInstances[i];
                LocalFeatureData localFeatureData = new LocalFeatureData(dependencyInstance, this);
                int dependencyInstanceLength = dependencyInstance.getLength();
                int[] predictedHeads = dependencyInstance.getHeads();
                int[] predictedLabels = new int[dependencyInstanceLength];

                localFeatureData.predictLabels(predictedHeads, predictedLabels, true);
                int numberCorrectMatches = getNumberCorrectMatches(dependencyInstance.getHeads(),
                        dependencyInstance.getDependencyLabelIds(),
                        predictedHeads, predictedLabels);
                if (numberCorrectMatches != dependencyInstanceLength - 1) {
                    loss += parameters.updateLabel(dependencyInstance, predictedHeads, predictedLabels,
                            localFeatureData, iIter * dependencyInstances.length + i + 1);
                }
                totalNUmberCorrectMatches += numberCorrectMatches;
                tot += dependencyInstanceLength - 1;
            }

            tot = tot == 0 ? 1 : tot;

            if (logger.isDebugEnabled()) {
                logger.debug(String.format("%n Iter %d loss=%.4f totalNUmberCorrectMatches=%.4f [%ds]%n",
                        iIter + 1, loss,
                        totalNUmberCorrectMatches / (tot + 0.0),
                        (System.currentTimeMillis() - start) / 1000));
            }

            parameters.printStat();
        }

    }

    private int getNumberCorrectMatches(int[] actualHeads, int[] actualLabels, int[] predictedHeads,
                                        int[] predictedLabels) {
        int nCorrect = 0;
        for (int i = 1, N = actualHeads.length; i < N; ++i) {
            if (actualHeads[i] == predictedHeads[i] && actualLabels[i] == predictedLabels[i])
                ++nCorrect;
        }
        return nCorrect;
    }

    DependencyLabel[] predictDependency(ConllData[][] document, String conllFormat) {

        ConllWriter conllWriter = new ConllWriter(options, dependencyPipe);

        DependencyLabel[] dependencyLabels = new DependencyLabel[document[0].length];

        for (ConllData[] sentence : document) {
            DependencyInstance dependencyInstance = dependencyPipe.nextSentence(sentence, conllFormat);
            if (dependencyInstance == null) {
                break;
            }
            LocalFeatureData localFeatureData = new LocalFeatureData(dependencyInstance, this);
            int numberOfTokensInSentence = dependencyInstance.getLength();
            int[] predictedHeads = dependencyInstance.getHeads();
            int[] predictedLabels = new int[numberOfTokensInSentence];
            localFeatureData.predictLabels(predictedHeads, predictedLabels, true);

            dependencyLabels = conllWriter.getDependencyLabels(dependencyInstance, predictedHeads, predictedLabels);
        }
        return dependencyLabels;
    }

}
