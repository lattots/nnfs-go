package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lattots/nnfs_go/pkg/neural_net"
	"github.com/lattots/nnfs_go/pkg/util"
)

const (
	learningDataFilename = "./data/mnist_train.csv"
	testingDataFilename  = "./data/mnist_test.csv"
)

func main() {
	trainSet, err := util.LoadDataset(learningDataFilename, 10)
	if err != nil {
		log.Fatal(err)
	}

	testSet, err := util.LoadDataset(testingDataFilename, 10)
	if err != nil {
		log.Fatal(err)
	}

	trainLoader := util.NewLoader(trainSet, 32, true)
	testLoader := util.NewLoader(testSet, 32, false)

	inputLayerConfig := &neuralnet.LayerConfig{InputSize: trainSet.Inputs.N, NeuronCount: 128, ActivationType: neuralnet.ReLU, InitType: neuralnet.He}
	firstHiddenConfig := &neuralnet.LayerConfig{NeuronCount: 64, ActivationType: neuralnet.ReLU, InitType: neuralnet.He}
	secondHiddenConfig := &neuralnet.LayerConfig{NeuronCount: 32, ActivationType: neuralnet.ReLU, InitType: neuralnet.He}
	outputLayerConfig := &neuralnet.LayerConfig{NeuronCount: 10, ActivationType: neuralnet.Softmax, InitType: neuralnet.He}

	nnConfig := neuralnet.Config{
		NumEpochs:    4,
		LearningRate: 0.1,
		LossFunction: neuralnet.CrossEntropy,
		LayerConfigs: []*neuralnet.LayerConfig{
			inputLayerConfig,
			firstHiddenConfig,
			secondHiddenConfig,
			outputLayerConfig,
		},
	}

	net, err := neuralnet.NewNet(nnConfig)
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	err = net.Train(trainLoader)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Training took", time.Since(start))

	successRate, err := net.Evaluate(testLoader)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Test complete with success rate of %.1f %%\n", successRate*100)
}
