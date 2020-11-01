from models import bbox_dispatcher as bbox
from models import lp_dispatcher as lp
import time
import tensorflow as tf
import joblib

BATCH_SIZE = 30


def measure_stage_2_latency():
    TARGET_SIZE = (280, 560)
    models = list(lp.CONSTRUCTORS.keys())
    timings = {}
    input_tensor = tf.random.uniform((BATCH_SIZE, ) + TARGET_SIZE + (3, ))
    for model_name in models:
        model = lp.create_model(model_name, TARGET_SIZE + (3, ))
        start = time.process_time_ns()
        model(input_tensor)
        end = time.process_time_ns()
        timings[model_name] = (end - start) / BATCH_SIZE
    print(timings)
    joblib.dump(timings, 'lp_timing.pkl')


def measure_stage_1_latency():
    TARGET_SIZE = (480, 480)
    models = list(bbox.CONSTRUCTORS.keys())
    timings = {}
    input_tensor = tf.random.uniform((BATCH_SIZE, ) + TARGET_SIZE + (3, ))
    for model_name in models:
        model = bbox.create_model(model_name, TARGET_SIZE + (3, ))
        start = time.process_time_ns()
        model(input_tensor)
        end = time.process_time_ns()
        timings[model_name] = (end - start) / BATCH_SIZE
    print(timings)
    joblib.dump(timings, 'bbox_timing.pkl')


if __name__ == '__main__':
    measure_stage_1_latency()
    measure_stage_2_latency()
