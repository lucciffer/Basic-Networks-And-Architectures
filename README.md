# Basic-Networks-And-Architectures  
### Networks covered in this repository 
- [Deep Convolutional GAN](DC-GAN)  
- [CNN based classifier for MNIST Data](CNN-based-MNIST-classifier)  
- [CNN based classifier for CIFAR10 Data](CNN-based-CIFAR10-classifier)
- [CNN based classifier for CIFAR100 Data](CNN-based-CIFAR100-classifier)
- [Convolutional Autoencoder on CIFAR10 Data](https://github.com/lucciffer/Basic-Networks-And-Architectures/tree/main/Convolutional-Autoencoder%20on%20CIFAR10)  


### Deep Convolutional Generative Adversarial Networks  
Released with the paper called [“Unsupervised Representation Learning with Deep Convolutional Adversarial Networks”](https://arxiv.org/abs/1511.06434v1) in 2016, the DCGAN is the state-of-the-art model. DCGAN is a GAN architecture that uses convolutions. The GAN basically consists of 2 main components. The generator, to generate data., and the discriminator to classify the images, and the both try to improve each other.  

**Generator:**  
As the name suggests, the generator is a network, that generates data, that could look either realistic or fake. The ultimate goal is to improve the generating ability of the generator to generate very realistic images/outputs by using the discriminator network. In this instance, we use the generator network to generate anime faces that we wish to look realistic.   
**Discriminator:**
Discriminator is the network that examines the performance of the Generator Network. Then again, How will this discriminator know the benchmarks for identifying real anime-like images from fake images that are produced by the generator? Even though the discriminator has no clue to classify the real anime images from the generated images at the beginning, It learns what are the benchmarks for examining the generator with the time with the help of the generator. Therefore it is basically learning how to classify generated images and the real images using the generated images.  
DCGAN is using the same concept with the help of convolution layers which are ideal for capturing patterns of images. The generator network uses a random noise in order to produce the image.  
**The architecture as published in the paper is as follows:** 
<img src="DC-GAN/assets/dcgan-arch.png">   
The architecture basically consists of convolution layers which typically help in capturing the details of the images.  
**There are 5 major key points that make DCGAN different from conventional GANs, they are as follows:**  
- No spatial pooling 
- No fully connected layers
- Batch Normalization 
- ReLU activation function for the Generator network
- Leaky ReLU activation function for the discriminator network  

