package com.company;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MainFaces {

    public static final Logger log = Logger.getAnonymousLogger();

    static final String LAYER_NAME = "avg_pool";

    static NativeImageLoader imageLoader = new NativeImageLoader(224  , 224, 3);

    static List<INDArray> vectors = new ArrayList<>();

    public static void main(String[] args) throws IOException {

        String faceFolder = "/home/yevhen/Downloads/face_detection_data";

        ZooModel zooModel = ResNet50.builder().numClasses(100).build();
        zooModel.initPretrained(PretrainedType.IMAGENET);
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained();

        log.info(pretrainedNet.summary());

        for (String facesPath : getFilePathes(faceFolder, 10)) {

            //All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
            // where H and W are expected to be atleast 224.

            INDArray indArray =  imageLoader.asMatrix(new File(facesPath));

            //The images have to be loaded in to a range of [0, 1]
            // and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(indArray);

            // todo normalize

            Map<String, INDArray> layersOutputs = pretrainedNet.feedForward(indArray, false);

            INDArray outLayer = layersOutputs.get(LAYER_NAME);

            vectors.add(outLayer);
        }
    }

    public static List<String> getFilePathes(String rootDir, int limit) throws IOException {
        try (Stream<Path> paths = Files.walk(Paths.get(rootDir))) {
            return paths.filter(Files::isRegularFile)
                        .map(p -> p.toFile().getAbsolutePath())
                        .limit(limit)
                        .collect(Collectors.toList());
        }
    }
}
