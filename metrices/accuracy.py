import tensorflow as tf

def Accuracy(model, dataset):
    accuracy = []
    count = 0
    for IMG, Y in  dataset:
        Y_hat = model(IMG)
        m = tf.keras.metrics.Accuracy()
        m.update_state(Y,Y_hat)
        accuracy.append(m.result().numpy())
        count += 1
    acc = sum(accuracy) / count
    return acc