# Perceptron image classifier

## Overview

A [perceptron](https://en.wikipedia.org/wiki/Perceptron) machine learning algorithm written in Rust. The perceptron can distinguish between 2 different shapes in an image. The training data is generated on the fly with a seeded rng. Also, a sigmoid function is used for the activation as I found it to produce slightly more accurate predictions.

## Model Training

![Model training steps](https://github.com/fakelag/perceptron-classifier/blob/master/training/train.gif)

Training is done in epochs configured in `TRAIN_EPOCHS` in `src/perceptron/config.rs`. Each epoch contains `TRAIN_STEP` number of iterations, each iteration containing a prediction for both shapes (and weight adjustment if the prediction is incorrect).

## Dataset

For different training and validation datasets, a seeded random is used when generating the shapes. The training seed is configured with `TRAIN_SEED` and validation with `VALIDATE_SEED`. The same seed always generates the same dataset.

Currently available shapes to be generated are RECT, TRIANGLE and CIRCLE.

```rust
pub const SHAPE_A: &str = "RECT";
pub const SHAPE_B: &str = "TRIANGLE";
```

#### Dataset A (sample)

![rectangle 1](https://github.com/fakelag/perceptron-classifier/blob/master/training/RECT_1.jpg)
![rectangle 2](https://github.com/fakelag/perceptron-classifier/blob/master/training/RECT_2.jpg)
![rectangle 3](https://github.com/fakelag/perceptron-classifier/blob/master/training/RECT_3.jpg)
![rectangle 4](https://github.com/fakelag/perceptron-classifier/blob/master/training/RECT_4.jpg)

#### Dataset B (sample)

![triangle 1](https://github.com/fakelag/perceptron-classifier/blob/master/training/TRIANGLE_1.jpg)
![triangle 2](https://github.com/fakelag/perceptron-classifier/blob/master/training/TRIANGLE_2.jpg)
![triangle 3](https://github.com/fakelag/perceptron-classifier/blob/master/training/TRIANGLE_3.jpg)
![triangle 4](https://github.com/fakelag/perceptron-classifier/blob/master/training/TRIANGLE_4.jpg)

## Snapshots

Enable `IMAGE_EPOCH_SNAPSHOTS` to take snapshots while training. Snapshot images will be stored in `training/`. The snapshot frequency scales with current epoch number.

![epoch 0](https://github.com/fakelag/perceptron-classifier/blob/master/training/epoch_0.jpg)
![epoch 5](https://github.com/fakelag/perceptron-classifier/blob/master/training/epoch_5.jpg)
![epoch 10](https://github.com/fakelag/perceptron-classifier/blob/master/training/epoch_10.jpg)
![epoch 100](https://github.com/fakelag/perceptron-classifier/blob/master/training/epoch_100.jpg)
![epoch final](https://github.com/fakelag/perceptron-classifier/blob/master/training/final.jpg)