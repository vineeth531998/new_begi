import pandas as pd
import yaml
import os
from sklearn.preprocessing import StandardScaler

def data_preprocess(tra_data_path,test_data_path):
   train=pd.read_csv(tra_data_path)
   test=pd.read_csv(test_data_path)
   scaler=StandardScaler()
   train_final=scaler.fit_transform(train)
   test_final=scaler.transform(test)
   return train, test

if __name__=="__main__":
    param_yaml_path="params.yaml"
    with open(param_yaml_path) as yaml_file:
        param_yaml = yaml.safe_load(yaml_file)
    train_data_path=os.path.join(param_yaml['split']['dir'],param_yaml['split']['train_file'])
    test_data_path=os.path.join(param_yaml['split']['dir'],param_yaml['split']['test_file'])
    final_train_data,final_test_data=data_preprocess(train_data_path,test_data_path)
    os.makedirs(param_yaml['process']['dir'],exist_ok=True)
    train_data_path=os.path.join(param_yaml['process']['dir'],param_yaml['process']['train_file'])
    final_train_data.to_csv(train_data_path,index=False)
    test_data_path=os.path.join(param_yaml['process']['dir'],param_yaml['process']['test_file'])
    final_test_data.to_csv(test_data_path,index=False)
