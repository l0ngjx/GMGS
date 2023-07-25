# GMGS

## Environment
* Python=3.7
* PyTorch=1.10.1
* numpy=1.20.3
## Usage
### Datasets

Our data has been preprocessed and is available at https://www.dropbox.com/sh/h548vcds8a4m3qs/AABH-YavkkoNFPR_RMTtmILOa?dl=0. You need to download the *datasets* folder and put it under the root. All the original dataset files are available at https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AAAMMlmNKL-wAAYK8QWyL9MEa/Datasets?dl=0&subfolder_nav_tracking=1.

* Diginetica dataset has a category attribute. Its attribute information can be found at https://competitions.codalab.org/competitions/11161#learn_the_details-data2.
* 30music dataset has an artist attribute. Each item's attribute value can be found in the original dataset file.
* The brief version of the Tmall dataset is also available on that website. And you can find its corresponding original file containing attribute information at https://tianchi.aliyun.com/dataset/dataDetail?dataId=42. It includes a category and a brand attribute.

### Train and test

Train and evaluate the model:

```sh
python main.py --dataset Tmall
```

