from sklearn.ensemble import RandomForestClassifier
import yaml
import os
import pandas as pd
import pickle
import numpy as np

def train(param_yaml_path):
    with open(param_yaml_path) as yaml_file:
        param_yaml = yaml.safe_load(yaml_file)
    depth=param_yaml['train']['n_est']
    random_state=param_yaml['base']['random_state']
    prediction_column=[param_yaml['base']['target_col']]
    clf=RandomForestClassifier(max_depth=depth,random_state=random_state)
    train_data_path=os.path.join(param_yaml['process']['dir'],param_yaml['process']['train_file'])
    train_df=pd.read_csv(train_data_path)
    x_train=train_df.drop(prediction_column,axis=1)
    y_train=np.squeeze(train_df[prediction_column])
    test_data_path=os.path.join(param_yaml['process']['dir'],param_yaml['process']['test_file'])
    test_df=pd.read_csv(test_data_path)
    x_test=test_df.drop(prediction_column,axis=1)
    y_test=np.squeeze(test_df[prediction_column])
    clf.fit(x_train,y_train.ravel())
    model_dir=param_yaml['model_dir']
    os.makedirs(model_dir,exist_ok=True)
    with open(model_dir+"/model.pkl","wb") as f:
        pickle.dump(clf,f)


if __name__=="__main__":
    train(param_yaml_path='params.yaml')