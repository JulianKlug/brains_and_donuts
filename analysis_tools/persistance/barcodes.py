import numpy as np
import gudhi

def barcode(filt):
    
    cplx = gudhi.CubicalComplex(dimensions = filt.shape, top_dimensional_cells = filt.flatten(order="F"))
    pd = cplx.persistence(homology_coeff_field=2, min_persistence=0)
    
    return pd

def barcode_plot(pd):
    plt = gudhi.plot_persistence_diagram(persistence=pd, legend=True)
    plt.show()
    
def separate_barcode(pd): 
    
    pd_dim0 = []
    pd_dim1 = []
    pd_dim2 = []
   
    for i in range(len(pd)):
        if pd[i][0] == 0:
            pd_dim0.append(pd[i][1])
            
        if pd[i][0] == 1:
            pd_dim1.append(pd[i][1])
        
        if pd[i][0] == 2:
            pd_dim2.append(pd[i][1])
    
    
    list_pd = [pd_dim0, pd_dim1, pd_dim2]
    
    return list_pd