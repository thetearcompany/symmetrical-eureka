export interface Tensor {
    shape: number[];
    data: Float32Array;
    reshape(newShape: number[]): Tensor;
    add(other: Tensor): Tensor;
    multiply(other: Tensor): Tensor;
    dot(other: Tensor): Tensor;
    softmax(): Tensor;
    relu(): Tensor;
}

export class TensorImpl implements Tensor {
    shape: number[];
    data: Float32Array;
    
    constructor(shape: number[], data?: Float32Array) {
        this.shape = shape;
        this.data = data || new Float32Array(shape.reduce((a, b) => a * b, 1));
    }

    reshape(newShape: number[]): Tensor {
        return new TensorImpl(newShape, this.data);
    }

    add(other: Tensor): Tensor {
        let result = new Float32Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) result[i] = this.data[i] + other.data[i];
        return new TensorImpl(this.shape, result);
    }

    multiply(other: Tensor): Tensor {
        let result = new Float32Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) result[i] = this.data[i] * other.data[i];
        return new TensorImpl(this.shape, result);
    }

    dot(other: Tensor): Tensor {
        let result = new Float32Array(this.shape[0] * other.shape[1]);
        let outputShape = [this.shape[0], other.shape[1]];
        for (let i = 0; i < this.shape[0]; i++)
            for (let j = 0; j < other.shape[1]; j++) {
                let sum = 0;
                for (let k = 0; k < this.shape[1]; k++) sum += this.data[i * this.shape[1] + k] * other.data[k * other.shape[1] + j];
                result[i * other.shape[1] + j] = sum;
            }
        return new TensorImpl(outputShape, result);
    }

    softmax(): Tensor {
        let max = Math.max(...this.data);
        let expValues = this.data.map(v => Math.exp(v - max));
        let sum = expValues.reduce((a, b) => a + b, 0);
        let normalized = expValues.map(v => v / sum);
        return new TensorImpl(this.shape, new Float32Array(normalized));
    }

    relu(): Tensor {
        let result = new Float32Array(this.data.length);
        for (let i = 0; i < this.data.length; i++) result[i] = Math.max(0, this.data[i]);
        return new TensorImpl(this.shape, result);
    }
}

export interface Layer {
    forward(input: Tensor): Tensor;
    backward(error: Tensor): Tensor;
}

export class DenseLayer implements Layer {
    weights: Tensor;
    biases: Tensor;
    
    constructor(inputSize: number, outputSize: number) {
        this.weights = new TensorImpl([inputSize, outputSize], new Float32Array(inputSize * outputSize).map(() => Math.random() - 0.5));
        this.biases = new TensorImpl([1, outputSize], new Float32Array(outputSize).map(() => Math.random() - 0.5));
    }

    forward(input: Tensor): Tensor {
        return input.dot(this.weights).add(this.biases).relu();
    }

    backward(error: Tensor): Tensor {
        return error;
    }
}

export interface Loss {
    compute(output: Tensor, target: Tensor): number;
    gradient(output: Tensor, target: Tensor): Tensor;
}

export class CategoricalCrossEntropy implements Loss {
    compute(output: Tensor, target: Tensor): number {
        let probs = output.softmax();
        let loss = 0;
        for (let i = 0; i < target.data.length; i++) loss -= target.data[i] * Math.log(probs.data[i]);
        return loss;
    }

    gradient(output: Tensor, target: Tensor): Tensor {
        let probs = output.softmax();
        let grad = new Float32Array(output.data.length);
        for (let i = 0; i < output.data.length; i++) grad[i] = probs.data[i] - target.data[i];
        return new TensorImpl(output.shape, grad);
    }
}

export interface Optimizer {
    update(layer: DenseLayer, gradient: Tensor): void;
}

export class SGD implements Optimizer {
    learningRate: number;

    constructor(learningRate: number) {
        this.learningRate = learningRate;
    }

    update(layer: DenseLayer, gradient: Tensor): void {
        for (let i = 0; i < layer.weights.data.length; i++) layer.weights.data[i] -= this.learningRate * gradient.data[i];
        for (let i = 0; i < layer.biases.data.length; i++) layer.biases.data[i] -= this.learningRate * gradient.data[i];
    }
}

export class NeuralNetwork {
    layers: Layer[];
    lossFunction: Loss;
    optimizer: Optimizer;

    constructor(lossFunction: Loss, optimizer: Optimizer) {
        this.layers = [];
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
    }

    addLayer(layer: Layer): void {
        this.layers.push(layer);
    }

    predict(input: Tensor): Tensor {
        let output = input;
        for (let layer of this.layers) output = layer.forward(output);
        return output;
    }

    train(input: Tensor, target: Tensor, epochs: number): void {
        for (let i = 0; i < epochs; i++) {
            let output = this.predict(input);
            let loss = this.lossFunction.compute(output, target);
            let error = this.lossFunction.gradient(output, target);

            for (let layer of this.layers.reverse()) error = layer.backward(error);

            this.optimizer.update(this.layers[0] as DenseLayer, error);

            console.log(`Epoch ${i + 1}: Loss = ${loss}`);
        }
    }
}

export class RPCServer {
    port: number;
    neuralNetwork: NeuralNetwork;

    constructor(port: number, neuralNetwork: NeuralNetwork) {
        this.port = port;
        this.neuralNetwork = neuralNetwork;
    }

    start(): void {
        console.log(`RPC Server running on port ${this.port}`);
    }

    handleRequest(request: { input: number[] }): { output: number[] } {
        let inputTensor = new TensorImpl([1, request.input.length], new Float32Array(request.input));
        let outputTensor = this.neuralNetwork.predict(inputTensor);
        return { output: Array.from(outputTensor.data) };
    }
}