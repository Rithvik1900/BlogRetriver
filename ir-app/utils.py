import numpy as np
from numpy.linalg import norm

def cosine_similarity(query,keys):
    p1 = query.dot(keys)
    p2 = norm(keys,axis=0)*norm(query)
    return p1/p2