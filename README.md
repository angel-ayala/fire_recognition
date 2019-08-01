# Lightweight and efficient octave convolutional neural network for fire recognition

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
