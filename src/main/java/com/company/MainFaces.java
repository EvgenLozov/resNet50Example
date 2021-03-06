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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MainFaces {

    public static final Logger log = Logger.getAnonymousLogger();

    static final String LAYER_NAME = "avg_pool";

    public static void main(String[] args) throws IOException {
        int limit = 8000;


        String faceFolder = "d:\\projects\\souteneur\\images";

        ZooModel zooModel = ResNet50.builder().numClasses(100).build();
        zooModel.initPretrained(PretrainedType.IMAGENET);
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained();

        log.info(pretrainedNet.summary());

        Function<String,INDArray> imageLoader = new NormalizedImageMatrixLoaderProvider().byFile();

        List<String> images = getFilePathes(faceFolder, limit);
        INDArray vectors = Nd4j.zeros(2048, limit);
        int index = 0;
        for (String facesPath : images) {

            //All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
            // where H and W are expected to be atleast 224.

            INDArray indArray =  imageLoader.apply(facesPath);

            //The images have to be loaded in to a range of [0, 1]
            // and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]


            Map<String, INDArray> layersOutputs = pretrainedNet.feedForward(indArray, false);

            INDArray outLayer = layersOutputs.get(LAYER_NAME).getRow(0).getColumn(0).getColumn(0);

            outLayer = outLayer.mul(1.0/outLayer.norm2(0).getDouble(0));


            vectors.putColumn(index++, outLayer);
        }

        SimilarImageService similarImageService = new SimilarImageService(vectors, images );

        similarImageService.find(0, 10);

    }

    private static void copyFilesToResultFolder(List<String> files) throws IOException {
        String folder = "result";

        deleteDirectoryRecursionJava6(new File(folder));

        new File(folder).mkdir();

        for (String file : files) {
            File source = new File(file);
            File destination = new File(folder+"/"+source.getName());
            copyFileUsingStream(source, destination);

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

    private static void copyFileUsingStream(File source, File dest) throws IOException {
        InputStream is = null;
        OutputStream os = null;
        try {
            is = new FileInputStream(source);
            os = new FileOutputStream(dest);
            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                os.write(buffer, 0, length);
            }
        } finally {
            is.close();
            os.close();
        }
    }

    private static void deleteDirectoryRecursionJava6(File file) throws IOException {
        if (file.isDirectory()) {
            File[] entries = file.listFiles();
            if (entries != null) {
                for (File entry : entries) {
                    deleteDirectoryRecursionJava6(entry);
                }
            }
        }
    }
}
