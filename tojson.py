import numpy as np
import pickle
import json
from scipy import sparse

class NumpyEncoder(json.JSONEncoder):
      def default(self, obj):
          if isinstance(obj, np.integer):
             return int(obj)
          elif isinstance(obj, np.floating):
             return float(obj)
          elif isinstance(obj, np.ndarray):
             return obj.tolist()
          elif isinstance(obj,sparse.csc_matrix):
             return obj.todense()
          else:
             return super(NumpyEncoder, self).default(obj)


if __name__ == '__main__':

    pklfile = input("input model pkl as ./basicModel_f_lbs_10_207_0_v1.0.0.pkl:")
    f0 = open( pklfile, 'rb')
    params=pickle.load(f0, encoding='latin1' ) #encoding='latin1'   encoding='iso-8859-1'
    f0.close()

    J_regressor = params['J_regressor']
    weights = params['weights']
    posedirs = params['posedirs']
    v_template = params['v_template']
    shapedirs = params['shapedirs']
    faces = params['f']
    kintree_table = params['kintree_table']
    #print( kintree_table )
    #print( shapedirs )
    #print(type(shapedirs))

    shape1=np.array(shapedirs,dtype=float)
    print(shape1.shape)

    data = {
        'kintree_table': kintree_table,
        'J_regressor': J_regressor,
        'v_template': v_template,
        'shapedirs': np.array(shapedirs,dtype=float),
        'posedirs': posedirs,
        'weights': weights,
        'f': faces
    }

    jsonfile = pklfile.replace('pkl', 'json')
    print( "write json: %s" %(jsonfile ) )

    f1=open( jsonfile,'w')
    json.dump( data, f1, cls=NumpyEncoder )
    f1.close()
    # f2=open('./model.json','r')
    # d=json.load(f2)