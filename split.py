import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

def data_split(param_yaml_path):
    with open(param_yaml_path) as yaml_file:
        param_yaml = yaml.safe_load(yaml_file)
    dataset = param_yaml['data_source']['local_path']
    df=pd.read_csv(dataset)
    #x=df.drop('price_range',axis=1)
    #y=df['price_range']
    split_ratio=param_yaml['split']['split_ratio']
    random_state=param_yaml['base']['random_state']
    train,test=train_test_split(df,test_size=split_ratio,random_state=random_state)

    os.makedirs(param_yaml['split']['dir'],exist_ok=True)
    train_data_path=os.path.join(param_yaml['split']['dir'],param_yaml['split']['train_file'])
    train.to_csv(train_data_path,index=False)
    test_data_path=os.path.join(param_yaml['split']['dir'],param_yaml['split']['test_file'])
    test.to_csv(test_data_path,index=False)

if __name__=="__main__":
    data_split(param_yaml_path="params.yaml")