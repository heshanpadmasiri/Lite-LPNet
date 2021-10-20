# Lite-LPNet
Model training and evalution codes for [Lite-LPNet](https://onlinelibrary.wiley.com/doi/10.1002/int.22471)

# Setting up dependencies
Code is written in python 3.8 and tensorflow 2.3. To install all python dependencies one may use the provided `requirements.txt` file. For GPU training we have used CUDA 10.1.
```
pip install -r requirements.txt
```
# Training models
Provided scripts assume you are using the [CCPD](https://github.com/detectRecog/CCPD) dataset. Annotation information (bounding box coordinates and license plate content) is extracted from the name of the image file (for more details about this naming scheme please refer to the original dataset). If you are using a different dataset as long as you maintain the same naming convention our dataset codes (`dataset/*`) should work with them as well.
## Bounding box models
To train individual bounding box models (stage 1) use the provided `train_bbox.py` script

```
python train_bbox.py PATH_TO_DATASET MODEL_NAME [input_shape] [restore_weights] [mixed_precision] [folds] [starting_fold] [cache_dataset]
```

Here PATH_TO_DATASET refers to the path to CCPD dataset. For availbel models please refer to `./models` directory. When providing the MODEL_NAME use `v{model_number}` (ex: for `bbox_16` use `v16`).

Default input size is 480 x 480. If mixed_precision is enabled we will use `mixed_float16` (We recomand using mixed precision for faster model training if your GPU support it). If fold variable is not set we will use a train, test, validation split of 70:15:15 each. Training logs (We are using tensorboard) will be written by default to `./logs/simple_bbox/{model_name}/`. Training will use early stopping if the validation dataset's iou haven't improved within the last 10 iterations. We will also automatically save model weights to `./checkpoints/{model_name}`. (If you wish to change these callback functions please modify the `__get_callbacks__` function). At the end of training model will be saved to `saved_models/simple_bbox/{model_name}`

## License plate recognition models
To train the license plate recognition model (stage 2) use the provided `train_lp.py` script. You have to train seperate models for each individual character seperatly. (For more details please refer to our paper). 

```
python train_lp.py PATH_TO_DATASET MODEL_NAME IDX [input_shape] [restore_weights]
```

For these models we always use mixed precision. If your GPU don't suppor this please manually change this in the script. If you don't wish to change the default parameter you can also used the `train_lp_all.sh` to train models for each character in one go. All paramters have the same meaning as bounding box model. IDX stands for which character you need to train the model for. We will write logs to `./logs/lp_seperate/{model_name}/{idx}` directory and checkpoints to `./checkpoints/{model_name}/` directory. At the end of training model will be saved to `saved_models/lp_seperate/{model_name}/{idx}`.

In order to generate the final combined model use the `load_model_lp` provided under `model_utils.py`

# Converting to Tensorflow lite models

Helper function provided under `create_tflite_model.py`

# Evaluating models
Metrices for evaluating both stage 1 and 2 models are provided under `./metrices/` directory. These metrices are used in the callbacks used both for logging and early stopping