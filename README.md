# Lightweight and efficient octave convolutional neural network for fire recognition

Fire recognition from visual scenes is a demanding task due to the high variance of color and texture.
In recent years, several fire-recognition approaches based on deep learning methods have been proposed to overcome this problem.
However, building deep convolutional neural networks usually involves hundreds of layers and thousands of channels, thus requiring excessive computational cost, and a considerable amount of data. 
Therefore, applying deep networks in real-world scenarios remains an open challenge, especially when using devices with limitations in hardware and computing power, e.g., robots or mobile devices.  
To address this challenge, in this paper, we propose a lightweight and efficient octave convolutional neural network for fire recognition in visual scenes.
Extensive experiments are conducted on FireSense, CairFire, FireNet, and FiSmo datasets. 
In overall, our architecture comprises fewer layers and fewer parameters in comparison with previously proposed architectures. 
Experimental results show that our model achieves higher accuracy recognition, in comparison to state-of-the-art methods, for all tested datasets.

This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001, the Brazilian agencies FACEPE, CNPq, and Universidad Central de Chile under the research project CIP2018009.

---

> If you are interested in fire recognition models maybe would like to visit my [KutralNet Model](https://github.com/angel-ayala/kutralnet)

This code is the results of a proposed model using ResNet and Octave Convolution for fire recognition.
This was based on the octave implementation published in the [titu1994 repo](https://github.com/titu1994/keras-octconv) and modified for this purpose.

Also a baseline cross-dataset validation and testing is proposed with 4 differents datasets:

* FireSense: [FIRESENSE database of videos for flame and smoke detection](https://zenodo.org/record/836749)
* CairFire: [Fire-Detection-Image-Dataset](https://github.com/cair/Fire-Detection-Image-Dataset)
* FireNet: used for another work [link here](https://github.com/arpit-jadon/FireNet-LightWeight-Network-for-Fire-Detection), with a [Google Drive folder](https://drive.google.com/drive/folders/1HznoBFEd6yjaLFlSmkUGARwCUzzG4whq) public available
* FiSmo: A Compilation of Datasets fromEmergency Situations for Fire and Smoke Analysis, with a public dataset published in their [github repo](https://github.com/mtcazzolato/dsw2017).

Thanks to the authors who recopiled those datasets.

More details will be posted next.

A detailed version of the model
<img src="https://github.com/angel-ayala/fire_recognition/blob/master/models/OctFiResNet_model.png?raw=true" height=100% width=50%>
