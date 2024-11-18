# CMNIST & 10CMNIST


![image](https://github.com/Mohamad-Ghodrati/Colored-MNIST-Datasets/blob/main/images/10CMNIST.jpg?raw=true)  
This repository provides **CMNIST** and **10CMNIST** datasets. The implementation draws from two key papers:

- **CMNIST**: Derived from the paper *Invariant Risk Minimization* by Arjovsky et al., this dataset simulates environmental biases in the MNIST dataset to evaluate invariant learning methods. ([Paper](https://arxiv.org/abs/1907.02893), [GitHub](https://github.com/facebookresearch/InvariantRiskMinimization))
  
- **10CMNIST**: Based on the paper *Towards Environment Invariant Representation Learning*, this dataset extends the CMNIST concept to ten distinct environments. ([Paper](https://openreview.net/forum?id=c4l4HoM2AFf))


## Dataset Preparation
The datasets are generated from the original MNIST dataset. Use the provided scripts to download and preprocess the data automatically.

## Usage

### Example Workflow


Open the Jupyter [notebook](notebooks/dataset_usage_demo.ipynb) in `notebooks/dataset_usage_demo.ipynb` to explore the datasets:
   
   

## Results

### CMNIST
![image](https://github.com/Mohamad-Ghodrati/Colored-MNIST-Datasets/blob/main/images/CMNISTSample.png?raw=true)

### 10CMNIST (Downsampled)
![image](https://github.com/Mohamad-Ghodrati/Colored-MNIST-Datasets/blob/main/images/10CMNISTSample.png?raw=true)


## Acknowledgments

I acknowledge the authors of:
- *Invariant Risk Minimization* ([Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
- *Towards Environment Invariant Representation Learning* ([Eyre et al., 2022](https://openreview.net/forum?id=c4l4HoM2AFf))


## Contributing

Contributions, bug fixes, and feature requests are welcome! Please open an issue or submit a pull request.
