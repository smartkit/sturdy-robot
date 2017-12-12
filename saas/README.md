# sturdy-robot
smartkit.ai

```
----------------------------------------------------------
# split data
data = ...
train, validation, test = split(data)

# tune model hyperparameters
parameters = ...
for params in parameters:
	model = fit(train, params)
	skill = evaluate(model, validation)

# evaluate final model for comparison with other models
model = fit(train)
skill = evaluate(model, test)
----------------------------------------------------------
```

### Reference

#### DICOM

http://dicom.nema.org/

https://en.wikipedia.org/wiki/Picture_archiving_and_communication_system

https://github.com/FNNDSC/pypx

#### DeepLearning

Caffe-SegNet: http://mi.eng.cam.ac.uk/projects/segnet/

Bayesian-SegNet: https://github.com/anlthms/whale-2015

Caffe-ENet:https://github.com/TimoSaemann/ENet

Tensorflow-Unet: https://github.com/jakeret/tf_unet
