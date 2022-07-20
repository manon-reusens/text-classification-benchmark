#Import packages
import pandas as pd
from datasets import load_dataset


def huggingface_to_csv(name, 
                       subset,
                       output_path):

    dataset = load_dataset(name,subset)

    if 'validation' in  dataset.keys():
        pd.DataFrame(dataset['train']).to_csv(output_path + 'train.csv',index=False)
        pd.DataFrame(dataset['validation']).to_csv(output_path + 'val.csv',index=False)
        pd.DataFrame(dataset['test']).to_csv(output_path + 'test.csv',index=False)
    else :
        pd.DataFrame(dataset['train']).to_csv(output_path + 'train.csv',index=False)
        pd.DataFrame(dataset['test']).to_csv(output_path + 'test.csv',index=False)

huggingface_to_csv('silicone','dyda_e','datasets/sentiment/emotion/silicone/')
huggingface_to_csv('liar','default','datasets/fake_news/liar/')
huggingface_to_csv('emotion','default','datasets/sentiment/emotion/CARER/')
huggingface_to_csv('imdb','plain_text','datasets/sentiment/emotion/IMDb/')
huggingface_to_csv('ag_news','default','datasets/topic/ag_news/')
huggingface_to_csv('yelp_polarity','plain_text','datasets/sentiment/polarity/yelp/')
huggingface_to_csv('sst2','default','datasets/sentiment/polarity/SST2/')