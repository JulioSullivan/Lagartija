from sklearn.externals import joblib
import pdb
loaded_model = joblib.load('trained_model_02.pkl')
loaded_idf = joblib.load('idf_model.pkl')
pdb.set_trace()