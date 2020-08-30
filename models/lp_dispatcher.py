from models.lp_2 import create_model as create_model_2

CONSTRUCTORS = {
    'v2' : create_model_2
}

def create_model(model_name):
    model_name = model_name.lower()
    return CONSTRUCTORS[model_name]()