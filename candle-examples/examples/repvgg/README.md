# candle-repvgg

A candle implementation of inference using a pre-trained [repvgg](https://arxiv.org/abs/2101.03697).
This uses a classification head trained on the ImageNet dataset and returns the
probabilities for the top-5 classes.

## Running an example

```
$ cargo run --example repvgg --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg

loaded image Tensor[dims 3, 224, 224; f32]
model built
mountain bike, all-terrain bike, off-roader: 61.70%
bicycle-built-for-two, tandem bicycle, tandem: 33.14%
unicycle, monocycle     : 4.88%
crash helmet            : 0.15%
moped                   : 0.04%

```
