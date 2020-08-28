from models.bbox_20 import create_model as create_model_20
from models.bbox_19 import create_model as create_model_19

CONSTRUCTORS = {
    'v19' : create_model_19,
    'v20' : create_model_20
}

def create_model(model_name,input_shape):
    model_name = model_name.lower()
    return CONSTRUCTORS[model_name](input_shape)