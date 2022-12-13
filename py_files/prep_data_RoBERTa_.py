import torch
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset

class prep_data_RoBERTa():
    
    def __init__(self, train, val, test):
        """
        This class prepares the BERT data to then use as an input for the model.
        This part is separate from the rest of the RoBERTa architecture as it is not required to redo all these steps for each iteration of the hyperparameter tuning.
        """

        self.train= train
        self.val= val
        self.test= test
        

        #loading in the BERT tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True)
        self.max_len= self.get_maxlen()

    def get_maxlen(self):
        max_len = 0


        # For every sentence...
        for sent in self.train.text:

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True, max_length=512, truncation=True) #as the model only supports lengths up to 512

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))

        print('Max sentence length: ', max_len)  

        #we chose not to use the following step to use memory as efficient as possible, as their are many experiments
        #max_len = pow(2, math.ceil(math.log(max_len)/math.log(2))) #round to next power of 2 in case their are longer sequences in the testset
        
        return max_len  

    def prepare(self, data):
        input_ids = []
        attention_masks = []
        token_type_ids=[]

        # For every sentence...
        for sent in data.text:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = self.max_len,  # Pad all sentences 
                                truncation=True, #necessary for if longest sentence is in test set
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_token_type_ids=True,
                                return_tensors = 'pt',     # Return pytorch tensors.
                                )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

            token_type_ids.append(encoded_dict["token_type_ids"])
        
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        int_tweetEval_train_label=[int(i) for i in data.label]
        labels = torch.tensor(int_tweetEval_train_label)

        dataset= TensorDataset(input_ids, attention_masks, labels)

        return dataset

    def prepare_dataset(self):
        train_data= self.prepare(self.train)
        val_data= self.prepare(self.val)
        test_data= self.prepare(self.test)
        num_labels=len(self.train.label.unique())

        return train_data, val_data, test_data, num_labels

    


    

 