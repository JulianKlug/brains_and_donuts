import numpy as np
from skimage.morphology import binary_dilation

def thickening_3D(I_in,s=None) : #I = image in 3D matrix form (boolean) black =1, white=0 s=maximal number of steps

    I = I_in.copy() 
    if s is None:
        s = max(I.shape)  
        
    I = I_in.copy()
    
    for k in range(s): #k is the value of entry in the filtration
         
        I += binary_dilation(I)
     
    a = I.max()
    I = a - I + 1
    
    I[I == 0] = s*3 #set all zero values (non visited) to be the highest value of the filtration multiplied by 3 (so it's "infinite" on the barcode)
        
    return I



def thickening_2D(I_in,s=None) : #I = image in matrix form (boolean) black =1, white=0, s=number of steps in the filtration
    I = I_in.copy() 
    if s is None:
        s = max(I.shape)
    
    
    previous = [] #list of previously visited voxels
    
    previous = np.where(I == 1) #set the list to be the black voxels
    
    for k in range(s): #k is the value of entry in the filtration
        
        
        previous = np.where(I == k + 1) #visited voxels at last step
        
        for l in range(len(previous[0])): 
            i = previous[0][l]
            j = previous[1][l]
        
        #update value of unvisited yet neibors of (i,j) to be one more step in the filtration 
        
            if i + 1 < I.shape[0] and I[i + 1,j] == 0:
                I[i + 1 , j] = k + 2
            
            if i - 1 >= 0 and I[i - 1,j] == 0:
                I[i - 1 , j] = k + 2
        
            if j + 1 < I.shape[1] and I[i,j + 1] == 0:
                I[i , j + 1] = k + 2
        
            if j - 1 >= 0 and I[i,j - 1] == 0:
                I[i , j - 1] = k + 2
                
    #change to dilation = cv2.dilate(img,kernel,iterations = 1)

    I[I == 0] = s*3 #set all zero values (non visited) to be the highest value of the filtration multiplied by 3 (so it's "infinite" on the barcode)
                
    return I



def height_2D(I_in, axe) : #I = image in matrix form (boolean) black =1, white=0, s=number of steps in the filtration
    I = I_in.copy() 
    
    list_indices = np.where(I == 1)
    
    max_direction = I.shape[axe]

    for k in range(len(list_indices[0])): 
        
        i = list_indices[0][k]
        j = list_indices[1][k]
        
        if axe == 0 :
            I[i,j] = i+1
        else : 
            I[i,j] = j+1
        
    
    a = I.max()
    I[I == 0] = a * 3 #set all zero values (non visited) to be the highest value of the filtration multiplied by 3 (so it's "infinite" on the barcode)
      #attention à normalize!!!! sinon pas comparable      taille max matrix    
    return I





def height_3D(I_in, axe) : #I = image in matrix form (boolean) black =1, white=0, s=number of steps in the filtration
    I = I_in.copy() 
    
    list_indices = np.where(I == 1)
    
    max_direction = I.shape[axe]

    for k in range(len(list_indices[0])): 
        
        i = list_indices[0][k]
        j = list_indices[1][k]
        l = list_indices[2][k]
        
        if axe == 0 :
            I[i,j,l] = i+1
        else : 
            if axe == 1 : 
                I[i,j,l] = j+1
            else :
                I[i,j,l] = l+1
        
    
    a = I.max()
    I[I == 0] = a * 3 #set all zero values (non visited) to be the highest value of the filtration multiplied by 3 (so it's "infinite" on the barcode)
                
    return I