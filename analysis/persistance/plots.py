#plots 

import gudhi
import matplotlib.pyplot as plt

def plot_proj(image):

    plt.imshow(image.max(0));
    plt.colorbar()
    plt.show()
    
def plot_layer(image, layer): 
    plt.imshow(image[layer]);
    plt.colorbar()
    plt.show()
    
def barcode(pd):
    plt = gudhi.plot_persistence_diagram(persistence=pd, legend=True)
    plt.show()