import pandas as pd
import openai
import os
from tqdm import tqdm
import json
from dotenv import load_dotenv
from datetime import datetime

from get_inclusion_exlclusion_response import generate_prompt, get_llm_response_for_single_trial
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
def compare_dates_df(df,column_name):
    df[column_name+"_mask"] = False
    for index,row in df.iterrows():
        try:
            if int(row[column_name].split("-")[0]) > 2024:
                df.loc[index,column_name+"_mask"] = True
            elif int(row[column_name].split("-")[0]) == 2024:
                if int(row[column_name].split("-")[1]) >= 10:
                    df.loc[index,column_name+"_mask"] = True
        except Exception as e:
            print (row[column_name])
    df = df[df[column_name+"_mask"] != False]
    return df

df = pd.read_csv("heart_disease_trials_final.csv")
print ("Total rows pre-cleaning:",len(df.index))
#Keeping trails that are recruiting
df = df[df['Study Status'] == "RECRUITING"]
#Removing trails that are completed.
df = df[df['Completion Date'] != ""]
df = compare_dates_df(df,"Completion Date")
print ("Total rows post-cleaning:",len(df.index))

df = df.iloc[:100,:]
# df = df.iloc[:100,:]
# print (df)
missed_files = []
results = []
for idx, trial in tqdm(df.iterrows()):
    try:
        prompt = generate_prompt(trial)
    except Exception as e:
        print ("Error in generate_prompt due to:",e)
        missed_files.append(trial)
        continue
    # Call OpenAI to generate the response
    try:
        response = get_llm_response_for_single_trial(prompt)
    except Exception as e:
        print ("Error in get_llm_response_for_single_trial due to:",e)
        missed_files.append(trial)
        continue

    # Store the result in a dictionary
    trial_result = {
        'NCT Number': trial['NCT Number'],
        'Study Title': trial['Study Title'],
        'Response': response
    }

    # Append the result to the results list
    results.append(trial_result)

# Save the results to a JSON file
with open('clinical_trials_responses.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)
# import pdb;pdb.set_trace()
missed_files.append(trial)
dict_list = [s.to_dict() for s in missed_files]
with open('missed_files.json', 'w') as json_file:
    json.dump(dict_list, json_file, indent=4)
