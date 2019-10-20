## Models:
* __vgg16__ - TBA,
* __unet_draft__ - TBA,
* __experiment_1__:
  * unregularized
  * no dropout in this model
* __experiment_2__: 
  * unregularized
  * no dropout in this model
  * more filters than in __experiment_1__
* __experiment_3__: 
  * unregularized
  * no dropout in this model
  * more filters than in __experiment_2__
* __experiment_4__: 
  * regularized
  * add dropout after the convolutional but not on the upsampling layers
* __experiment_5__: 
  * regularized
  * add dropout after the convolutional
  * add dropout also after the upsampling layers
* __experiment_6__: 
  * regularized
  * add dropout after the convolutional
  * add dropout also after the upsampling layers
  * more filters than in __experiment_5__
* __experiment_7__: 
  * regularized
  * add dropout after the convolutional
  * add dropout also after the upsampling layers
  * change dropout to 0.1
  * more filters than in __experiment_6__
* __fcn8__:
  * common architecture with pretrained weights based on vgg16