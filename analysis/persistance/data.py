import pickle

def save(data,name=None):
    if name is None:
        name = str(data)
    
    f = open(name,'wb')
    pickle.dump(data,f)
    f.close() 
        
        
def load(name):
    with open(str(name), 'wb') as f:
        return pickle.load(f)