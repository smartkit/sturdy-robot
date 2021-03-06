# SegNetCMR
A [Tensorflow](https://www.tensorflow.org/) implementation of [SegNet](https://mi.eng.cam.ac.uk/projects/segnet/) to segments CMR images

## Aims
1. A demonstration of a more complete Tensorflow program including saving state and resuming.
2. Provide an ready-to-go example of medical segmentation with sufficient training and validation data, in a usable format (PNGs).

## Requirements
1. Python 3.5: Best to use the [Conda](https://www.continuum.io/downloads) distribution
2. Tensorflow 0.12

### nvidia-GPU

sudo apt-get install libcupti-dev

sudo dpkg -i /home/mini/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
 
sudo apt-get update

sudo apt-get install cuda

sudo ldconfig /usr/local/cuda/lib64

source activate py35

pip install -r requirements.txt

## Todo
1. Add code to run on your own data (currently there is only the training code present)

## Running
Make sure you have conda and tensorflow installed

```commandline
conda install tensorflow
python
Python 3.5.2 | packaged by conda-forge | (default, Sep  8 2016, 14:36:38)
```
The git clone this repository
```commandline
git clone https://
```

And start the training from the folder
```commandline
cd /path/to/SegNetCMR

python setup.py install

sudo ldconfig /usr/local/cuda/lib64/

python train.py
```

And in another terminal window start tensorboard
```commandline
tensorboard --logdir ./Output
```
Then in your webbrowser go to [http://localhost:6006](http://localhost:6006)

## Training and test data
Many thanks to the [Sunnybrook Health Sciences Centre](http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/) for providing a set of CMR data with associated contours.
Unfortunately, in the latest release the filenames have become a little mangled, and don't match up with the contours.
I have gone through the files and matched them up; exported the DICOMS as PNGs and converted the list of coordinates of the contours to PNGs as well.

The first two sets of CMRs are included as training data, the last set as test data.



## With thanks to
[andreaazzini/segnet](https://github.com/andreaazzini/segnet): A Tensorflow SegNet translation

[pydicom](https://github.com/darcymason/pydicom): A pure python dicom library

[StackOverflow Tensorflow batch_norm thread](http://stackoverflow.com/questions/40081697/getting-low-test-accuracy-using-tensorflow-batch-norm-function)

[GitHub Tensorflow unpool thread](https://github.com/tensorflow/tensorflow/issues/2169)

## Issues and annoyances
1. The original SegNet uses max_pool_with_argmax, and requires an unpool_with_argmax. Unfortunately, Tensorflow does not provide an unpool_with_argmax. Fortunately there is code in the github thread above to make your own.
2. This version of unpool_with_argmax runs on the CPU not GPU so is a little slower.
3. Tensorflow does not provide a CPU version of max_pool_with_argmax, so if you don't have a GPU you are stuck using the maxpool version.
4. Tensorflow forgot to include a function for gradients for maxpoolwithargmax, so it is included at the bottom of train.py
5. The name mangling of the Sunnybrook CMR data.

## License
SegNetCMR: MIT license

SunnyBrook Cardiac Data: Public Domain

pydicom: MIT license

### Troubleshoots

https://www.google.ca/url?sa=t&rct=j&q=&esrc=s&source=web&cd=9&cad=rja&uact=8&ved=0ahUKEwiri9n9_czVAhWGOyYKHe6rDAoQFghXMAg&url=https%3A%2F%2Fhyunyoung2.github.io%2F2017%2F06%2F27%2FHow_To_Install_Tensorflow%2F&usg=AFQjCNG0DaVdvgLme8a70DsDAru8NWmMtQ

#### Tips

ImageMagick:

TIFF to PNG:

```
mogrify -background black -format png -depth 8  Data/Training/Images/cancer_subset00/*.tiff
```
SVG to PNG:

```
mogrify -background black -format png -depth 8 Data/Training/Labels/cancer_subset00/*.svg
```

Resize:

```
mogrify -resize 50% Data/Training/Images/cancer_subset00/*.png
```

GrayScale

```
for file in Data/Training/Images/cancer_subset00/*.png; do convert $file  -colorspace Gray $file;done
```

SVG fill replace:

```
find ./ -type f -name '*.svg' | xargs -I{} sed -i_old -n -e 's/polygon fill="none"/polygon fill="white"/g;p;' {}
```

Gray to RGB

```
mogrify -type TrueColorMatte -define png:color-type=6  /Volumes/UUI/labels/normal/*.png

```
Rotate 90

```
mogrify -rotate 90 /Volumes/UUI/images/rotate90/*.png
```
Rename with prefix

```
for filename in *.png; do mv "$filename" "prefix_$filename"; done;
```
Flip

Flop



#### Train,Test results

| Train/Test Dataset |Train accuracy|Test accuracy|Train loss|Test loss|Epoch|
| -------------|-------------|------------- |-------------|-------------| -----:|
| 100/24 |0.966|0.649|0.388|0.687|1000|
| 100/24 |0.980|0.790|0.355|0.529|2000|
| 200/24 |0.937|0.779|0.461|0.579|1000|
| 200/24 |0.949|0.901|0.413|0.439|2000|
| 400/24 |0.910|0.865|0.466|0.467|2000|
| 536/60 |0.936|0.872|0.415|0.460|2000|
| 536/60 |0.964|0.861|0.343|0.443|5000|
| 536/60 |0.971|0.878|0.249|0.380|9000|
| 536/140 |0.886|0.996|0.479|0.359|1000|
| 536/140 |0.964|0.967|0.335|0.356|5000|
| 1072/140 |0.939|0.969|0.333|0.344|5000|
| 2144/140 |0.951|0.938|0.259|0.304|10000|

#### References

https://github.com/NVIDIA/DIGITS/tree/master/examples/medical-imaging

https://machinelearningmastery.com/difference-test-validation-datasets/

https://tensor-flow.com/validation-testing-model
