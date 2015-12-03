import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
import Plots
import pickle
srng = RandomStreams()

f = open("costs.txt", 'w')
f.write("Starting...\n")
f.close()


def write(str):
    f = open("costs.txt", 'a')
    f.write(str)
    f.write("\n")
    f.close()
    
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_biases(n_out):
    return theano.shared(floatX(np.zeros(n_out)))

def plotter(samples, predictions, img_x, idx):
    #plot_all_filters(Ws, idx)
    shp = (samples.shape[0], 1, img_x, img_x)
    samples = samples.reshape(shp)
    predictions = predictions.reshape(shp)
    Plots.plot_predictions_grid(samples, predictions, idx, shp, 'preds')
    return

def rectify(X):
    return T.maximum(X, 0.)

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w1, b1, wmu, bmu, wsigma, bsigma, w2, b2, w3, b3, e):
    h1 = rectify(T.dot(X, w1) + b1)
    mu = T.dot(h1, wmu) + bmu
    log_sigma = 0.5 * (T.dot(h1, wsigma) + bsigma)
    z = mu + T.exp(log_sigma) * e
    h2 = rectify(T.dot(z, w2) + b2)
    out = rectify(T.dot(h2, w3) + b3)
    return mu, log_sigma, z, out


def generate(Z, w2, b2, w3, b3):
    h2 = rectify(T.dot(Z, w2) + b2)
    out = rectify(T.dot(h2, w3) + b3)
    return out

    
# layers should be of the form
# (input size, n_hidden1, n_hidden_2, ..., input_size)
def get_params(code_size):
    n_hidden1 = 600
    n_code = code_size
    n_hidden2 = 600
    w1 = init_weights((784, n_hidden1))
    b1 = init_biases(n_hidden1)
    wmu = init_weights((n_hidden1, n_code))
    bmu = init_biases(n_code)
    wsigma = init_weights((n_hidden1, n_code))
    bsigma = init_biases(n_code)
    w2 = init_weights((n_code, n_hidden2))
    b2 = init_biases(n_hidden2)
    w3 = init_weights((n_hidden2, 784))
    b3 = init_biases(784)
    return w1, b1, wmu, bmu, wsigma, bsigma, w2, b2, w3, b3

trX, trY, teX, teY, channels, img_x = mnist(onehot=True)
trX = trX.reshape(trX.shape[0], 784)
teX = teX.reshape(teX.shape[0], 784)
X = T.fmatrix()
e = T.matrix()

n_code = 2

w1, b1, wmu, bmu, wsigma, bsigma, w2, b2, w3, b3 = get_params(n_code)

mu, log_sigma, z, out = model(X, w1, b1, wmu, bmu, wsigma, bsigma, w2, b2, w3, b3, e)
recon_cost = T.sum(T.sqr(X-out))
kl_cost = 0.5 * T.sum(1 + 2 * log_sigma - mu ** 2 - T.exp(2 * log_sigma))

cost = recon_cost - kl_cost


params = [w1, b1, wmu, bmu, wsigma, bsigma, w2, b2, w3, b3]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, e], outputs=cost, updates=updates, allow_input_downcast=True)

reconstruct = theano.function(inputs=[X, e], outputs=[out, cost], allow_input_downcast=True)

z = T.fmatrix()
fantasy = generate(z, w2, b2, w3, b3)
fantasize = theano.function(inputs=[z], outputs=fantasy, allow_input_downcast=True)    

for i in range(101):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        c = train(trX[start:end], floatX(np.random.randn(128, n_code)))
    
    # if i % 10 == 0:
    samples = teX[:10, :]        
    r, cost = reconstruct(samples, floatX(np.random.randn(10, n_code)))
    print(cost)
    write(str(i) + ": " + str(cost))
    plotter(samples, r, 28, i)
    
    
    samples = floatX(np.random.randn(20, n_code))
    f = fantasize(samples)
    f1 = f[:10, :]
    f2 = f[10:, :]
    #plotter(f1, f2, 28, i+9999)
    shp = (f1.shape[0], 1, 28, 28)
    f1 = f1.reshape(shp)
    f2 = f2.reshape(shp)
    Plots.plot_predictions_grid(f1, f2, i, shp, 'generated')
    
    
    #plotter(samples, f, 28, 999)    
#         print(cost)
#         
# pickle.dump( Ws, open( "Ws.p", "wb" ) )
# pickle.dump( bs, open( "bs.p", "wb" ) )
