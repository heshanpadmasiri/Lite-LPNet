from models.bbox_dispatcher import create_model, CONSTRUCTORS
import time
import tensorflow as tf
import joblib

TARGET_SIZE = (480, 480)
BATCH_SIZE = 30


def measure_stage_1_latency():
    models = list(CONSTRUCTORS.keys())
    timings = {}
    input_tensor = tf.random.uniform((BATCH_SIZE, ) + TARGET_SIZE + (3, ))
    for model_name in models:
        model = create_model(model_name, TARGET_SIZE + (3, ))
        start = time.process_time_ns()
        model(input_tensor)
        end = time.process_time_ns()
        timings[model_name] = end - start
    print(timings)
    joblib.dump(timings, 'bbox_timing.pkl')


if __name__ == '__main__':
    measure_stage_1_latency()
