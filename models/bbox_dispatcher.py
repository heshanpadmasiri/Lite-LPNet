from models.bbox_20 import create_model as create_model_20
from models.bbox_19 import create_model as create_model_19
from models.bbox_16 import create_model as create_model_16
from models.bbox_17 import create_model as create_model_17
from models.bbox_18 import create_model as create_model_18

CONSTRUCTORS = {
    'v16' : create_model_16,
    'v17' : create_model_17,
    'v18' : create_model_18,
    'v19' : create_model_19,
    'v20' : create_model_20
}

def create_model(model_name,input_shape):
    model_name = model_name.lower()
    return CONSTRUCTORS[model_name](input_shape)