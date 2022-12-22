# zero-to-hero
Create and deploy to production a simple neural network for Computer Vision


# Tools Used
* JAX Library for computing gradients, performing tensor operations and scheming the segmentation model
* Wandb for metrics and training tools
* MLflow for deploying and compiling the model for production
* Gradio for interactive user-experience platform within an online platform (Data-ICMC Website).


## Datasets to consider
* [LabelMe 12 50k](https://www.kaggle.com/datasets/dschettler8845/labelme-12-50k)
* [City-Scapes](https://www.cityscapes-dataset.com/dataset-overview/)
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)


## References
* [DSNet-Fast](https://www.researchgate.net/figure/The-architecture-of-fast-dense-segmentation-network-DSNet-fast-The-encoder-is_fig1_347180093)
* [DSNet](https://www.researchgate.net/figure/The-architecture-of-dense-segmentation-network-DSNet-The-encoder-is-a-fully-convolutional_fig1_347180092)


# First Model
The first model available and deployed considers a simple 2D image, taken from the MNIST Dataset. Reefer to [_High-Performance Neural Networks
for Visual Object Classification_](https://arxiv.org/pdf/1102.0183.pdf) for further details on the dataset.

The CI/CD process will use the default Github pipeline using the available [Github Actions features](https://github.blog/2022-02-02-build-ci-cd-pipeline-github-actions-four-steps/). The training process will use the MLFLow framework, to cather and track the necessary metrics and log accordingly. Reefer to the [docs](https://mlflow.org/docs/latest/quickstart.html) for further details.
