import numpy
import numpy as np
def safe_softmax(input , axis = -1):
    z = input - np.max(input , axis = axis , keepdims=True)
    num = np.exp(z)
    output = num / np.sum(num , axis=axis , keepdims=True)
    return output

def cross_entropy_loss(y_true , y_pred , axis=-1):
    eps = 1e-15
    y_pred = np.clip(y_pred , eps , 1-eps)
    y_prob = -np.sum(y_true * np.log(y_pred) , axis = axis).mean()
    return y_prob

def softmax_cross_entropy_forward_backward(logits , y_true , batch_size , axis = -1):
    y_pred = safe_softmax(logits , axis=axis)
    loss = cross_entropy_loss(y_true , y_pred , axis)
    return loss , (y_pred - y_true) / batch_size