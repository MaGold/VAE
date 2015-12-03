import numpy as np
import os
#from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

#---------------------------------------------------------------------
def plot_filters(w, channels, idx, title):
    if channels == 1:
        plot_grey_filters(w, idx, title)
    elif channels == 3:
        plot_color_filters(w, idx, title)

def plot_grey_filters(x, idx, title=""):
    num_filters = x.shape[0]
    numrows = 10
    numcols = int(np.ceil(num_filters/10))
    plt.figure(figsize=(numrows, numcols))
    gs = gridspec.GridSpec(numcols, numrows)
    gs.update(wspace=0.1)
    gs.update(hspace=0.0)
    
    print("Plotting filters...")
    print(x.shape)
    for i in range(num_filters):
        ax = plt.subplot(gs[i])
        w = x[i, :, :, :]
        w = np.swapaxes(w, 0, 1)
        w = np.swapaxes(w, 1, 2)
        ax.imshow(w[:, :, 0], cmap=plt.cm.gist_yarg,
                  interpolation='nearest', aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
    plt.savefig(os.path.join('convfilters', title + '_' + str(idx) + '_convcaustics.png'))
    plt.close('all')

def plot_color_filters(x, idx, title=""):
    num_filters = x.shape[0]
    numrows = 10
    numcols = int(np.ceil(num_filters/10))
    plt.figure(figsize=(numrows, numcols))
    gs = gridspec.GridSpec(numcols, numrows)
    gs.update(wspace=0.1)
    gs.update(hspace=0.0)

    print("plotting color filters")
    for i in range(num_filters):
        ax = plt.subplot(gs[i])
        w = x[i, :, :, :]
        w = np.swapaxes(w, 0, 1)
        w = np.swapaxes(w, 1, 2)

        #normalize
        
        r = w[:,:,0] - np.min(w[:,:,0])
        g = w[:,:,1] - np.min(w[:,:,1])
        b = w[:,:,2] - np.min(w[:,:,2])
        r = r * 1.0 / np.max(r)
        g = g * 1.0 / np.max(g)
        b = b * 1.0 / np.max(b)
        w = np.dstack((r,g,b))
        ax.imshow(w,
                  cmap=plt.cm.gist_yarg,
                  interpolation='nearest',
                  aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
    plt.savefig(os.path.join('convfilters', title+ '_' + str(idx) + '_convcaustics.png'))
    print(os.path.join('convfilters', title+ '_' + str(idx) + '_convcaustics.png'))
    plt.close('all')

#--------------------------------------------------------------------------------------------
def plot_predictions(samples, predictions, k, batch_size, imgshape):
    samples = samples.reshape(batch_size, 3, imgshape, imgshape)
    predictions = predictions.reshape(batch_size, 3, imgshape, imgshape)
    samples = np.swapaxes(samples, 1,2)
    samples = np.swapaxes(samples, 2,3)
    predictions = np.swapaxes(predictions, 1,2)
    predictions = np.swapaxes(predictions, 2,3)
    for i in range(samples.shape[0]):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(samples[i, :, :, :],
                cmap=plt.cm.gist_yarg, interpolation='nearest',
                aspect='equal')
        path = "convpredictions"
        fname = str(k) + '_' + str(i) + 'sample_IN.png'
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(os.path.join(path, fname), dpi=imgshape)
        plt.close('all')
        fig = plt.figure(figsize=(5, 5))
        print(predictions[i, :, :, :].shape)
        print(predictions[i, :, :, :])
        plt.imshow(predictions[i, :, :, :],
                cmap=plt.cm.gist_yarg, interpolation='nearest',
                aspect='equal')
        fname = str(k) + '_' + str(i) + 'sample_OUT.png'
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(os.path.join(path, fname), dpi=imgshape)
        plt.close('all')

#--------------------------------------------------------------------
def plot_volume(volumes, k, title):
    for ind in [0,4]:
        vols = volumes[ind,:]
        vols = (vols-np.min(vols))/(np.max(vols)-np.min(vols))
        print(volumes.shape)
        numrows = 10
        numcols = int(np.ceil(vols.shape[0]/10))
        plt.figure(figsize=(numrows, numcols))
        gs = gridspec.GridSpec(numcols, numrows)
        gs.update(wspace=0.1)
        gs.update(hspace=0.0)
        
        for i in range(vols.shape[0]):
            ax = plt.subplot(gs[i])
            if i < vols.shape[0]:
                w = vols[i,:]
                
                ax.imshow(w,
                        cmap=plt.cm.gist_yarg,
                        interpolation='nearest',
                        aspect='equal')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis('off')
        if ind==0:
            plt.savefig(os.path.join('convvolumes', title + '_' + str(k) + '_' + '_vol_A.png'))
        else:
            plt.savefig(os.path.join('convvolumes', title + '_' + str(k) + '_' + '_vol_B.png'))
        plt.close('all')

#--------------------------------------------------------------------
def plot_predictions_grid(samples, predictions, k, imgshape, title=""):
    if imgshape[1] == 1:
        predictions_grid_grey(samples, predictions, k, imgshape, title)
    elif imgshape[1] == 3:
        predictions_grid_color(samples, predictions, k, imgshape, title)
    return

def predictions_grid_color(samples, predictions, k, imgshape, title):
    batch_size = samples.shape[0]
    print("printintg predictions:")
    print(samples.shape)
    print(imgshape)
    print(predictions.shape)
    samples = samples.reshape(batch_size, imgshape[1], imgshape[2], imgshape[3])
    predictions = predictions.reshape(batch_size, imgshape[1], imgshape[2], imgshape[3])
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 2)
    for i in range(10):
        ax = plt.subplot(gs[i])
        if i % 2 == 0:
            w = samples[i/2, :, :, :]
        else:
            w = predictions[i/2, :, :, :]
        w = np.swapaxes(w, 0, 1)
        w = np.swapaxes(w, 1, 2)
        ax.imshow(w,
                  cmap=plt.cm.gist_yarg,
                  interpolation='nearest',
                  aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
    gs.update(wspace=0)
    plt.savefig(os.path.join('convpredictions', str(k) + '_' + title + '.png'),
                bbox_inches='tight')
    plt.close('all')



def predictions_grid_grey(samples, predictions, idx, imgshape, title):
    batch_size = samples.shape[0]
    samples = samples.reshape(batch_size, imgshape[2], imgshape[3])
    predictions = predictions.reshape(batch_size, imgshape[2], imgshape[3])
    plt.figure(figsize=(6, 10))
    gs = gridspec.GridSpec(6, 2)
    for i in range(12):
        ax = plt.subplot(gs[i])
        if i % 2 == 0:
            w = samples[i/2, :, :]
        else:
            w = predictions[i/2, :, :]
            
        ax.imshow(w,
                  cmap=plt.cm.gist_yarg,
                  interpolation='nearest',
                  aspect='equal',
                  vmin=0, vmax=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    plt.savefig(os.path.join('convpredictions', str(idx) + '_' + title + '.png'),
                bbox_inches='tight')
    plt.close('all')

    
#-------------------------------------------------------------------

    
def plot_costs(costs):
    #print(costs)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.plot(np.log(costs))
    plt.savefig('convcosts.png')
    plt.close('all')


def plot_testimg(imgs):
    for i in range(np.min([imgs.shape[0], 10])):
        img = imgs[i, :]
        #img = img.reshape(ch28, 28)
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(img.shape)
        ax.imshow(img, cmap=plt.cm.gist_yarg)#, interpolation='nearest')
    # cmap=plt.cm.gist_yarg,
    # interpolation='nearest',
    # aspect='equal')
        plt.savefig(str(i) + '_testimg.png')
        #plt.savefig(os.path.join('fanta', str(i) + '_testimg.png'))
        plt.close()
