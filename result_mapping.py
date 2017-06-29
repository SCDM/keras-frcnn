import csv
import numpy as np
import pandas as pd
import sys
import re

def df_mapping(unique_input_dictionary):
    df = pd.read_csv('BHC_Key.csv')
    df.columns = ['a'] 
    # cleaning the data frame:
    df['a'] = df['a'].replace({"BHC        Consolidated     ":""},regex=True)
    df['a'] = df['a'].replace({"BHC        Unconsolidated     ":""},regex=True)
    df['a'] = df['a'].replace({" ":""},regex=True)
    ##df['a']
    key_vector = df['a'].replace({"\(.*$|-.*$|B.*$":""},regex=True)
    ##key_vector
    # removing the matching key number from the formula:
    temp_vec=[]
    for i in range(0,df.shape[0]):
            temp_str = df['a'][i]
            temp_vec.append(temp_str[len(key_vector[i]):len(temp_str)])
    df['a'] = temp_vec
    # removing: ()
    df['a'] = df['a'].replace({"\\(|\\)":""},regex=True)
    # remove the outlier (line 227):
    #df.drop(227, inplace=True)
    df['a'][227]='NA'
    df.reset_index
    #key_vector.drop(227, inplace=True)
    key_vector[227]='NA'
    key_vector.reset_index

    # create a unique 4 charcaters dataframe:
    # attach a random value to key_vecotr for training purposes 
    keys = []
    df_temp = df.copy()
    df_temp['a'] = df_temp['a'].replace({"\\+":""},regex=True)
    df_temp['a'] = df_temp['a'].replace({"\\-":""},regex=True)
    # create an empty vector and append the splited values to it:
    values = []
    df_temp = df_temp.reset_index()
    for i in range(0,df_temp.shape[0]):
            values.append(re.split('BH[0-9a-zA-Z][0-9a-zA-Z]',df_temp['a'][i]))
    flat_list = [item for sublist in values for item in sublist]
    flat_list = set(flat_list)
    flat_list = pd.DataFrame(list(flat_list))
    temp_vec = pd.DataFrame(range(0,flat_list.shape[0]))
    #unique_input_dictionary = pd.concat([flat_list, temp_vec], axis=1) # input table of values
    #unique_input_dictionary

    #remove the first 4 charcater from the formulas df:
    df['a'] = df['a'].replace({"BH[a-zA-Z][a-zA-Z]":"A"},regex=True)

    # match this input to the formula, populate the formula 
    #list(unique_input_dictionary)
    #unique_input_dictionary.columns = ['key','value']
    #unique_input_dictionary = unique_input_dictionary.set_index('key')['value'].to_dict()
    #type(unique_input_dictionary)
    #unique_input_dictionary
    # convert the dict int to str:
    #unique_input_dictionary
    key_vector_with_formula = pd.concat([key_vector,df], axis=1) # input table of values

    # replace the values:
    # change dict objects to str:
    for key in unique_input_dictionary.keys():
        unique_input_dictionary[key] = str(unique_input_dictionary[key])
    # and replace withing the data frame:
    df['a'] = df['a'].replace(unique_input_dictionary,regex=True)
    # calculate the str formula:
    calculated_vector=[]
    
    for row in df['a']:
        if 'A' not in row and not bool(re.search('[a-zA-Z]', row)):
            calculated_vector.append(eval(row))
        else:
            calculated_vector.append('NA')
    
    calculated_vector = pd.DataFrame(calculated_vector)
    print(key_vector_with_formula.shape)
    key_vector_with_formula = pd.concat([key_vector_with_formula,calculated_vector], axis=1,ignore_index=True) # input table of values
    key_vector_with_formula.columns = ['account_id','calcs','account_asReported_amount']
    key_vector_with_formula = key_vector_with_formula[~key_vector_with_formula['account_asReported_amount'].isin(['NA'])]
    key_vector_with_formula.to_csv('data/result/Commerce_bank_page_002.csv', encoding='utf-8', index = False)
    #unique_input_dictionary
    #eval(df['a'][319])
    key_vector_with_formula.shape