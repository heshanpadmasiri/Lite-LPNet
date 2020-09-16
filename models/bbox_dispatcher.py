from models.bbox_16 import create_model as create_model_16
from models.bbox_17 import create_model as create_model_17
from models.bbox_18 import create_model as create_model_18
from models.bbox_19 import create_model as create_model_19
from models.bbox_20 import create_model as create_model_20
from models.bbox_21 import create_model as create_model_21
from models.bbox_22 import create_model as create_model_22
from models.bbox_23 import create_model as create_model_23
from models.bbox_24 import create_model as create_model_24
from models.bbox_25 import create_model as create_model_25
from models.bbox_26 import create_model as create_model_26
from models.bbox_27 import create_model as create_model_27
from models.bbox_28 import create_model as create_model_28

CONSTRUCTORS = {
    'v16': create_model_16,
    'v17': create_model_17,
    'v18': create_model_18,
    'v19': create_model_19,
    'v20': create_model_20,
    'v21': create_model_21,
    'v22': create_model_22,
    'v23': create_model_23,
    'v24': create_model_24,
    'v25': create_model_25,
    'v26': create_model_26,
    'v27': create_model_27,
    'v28': create_model_28
}


def create_model(model_name, input_shape):
    model_name = model_name.lower()
    return CONSTRUCTORS[model_name](input_shape)
