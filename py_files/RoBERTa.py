import time
import datetime
import numpy as np
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import Adam




class RoBERTa_model():
    
    def __init__(self,device,  config, train, val, test, num_labels, gpu=False):
        """
        The RoBERTa_model class will finetune the RoBERTa model according to the given parameters in the config.
        Next, the class will also make predictions on the test set.
        Note that when doing hyperparameter tuning, this class should be initiated within the loop
        
        config= the config file through wandb
        """
        #loading in the BERT tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation= True)

        self.device=device
        self.config= config
        self.train=train
        self.val= val
        self.test=test
        self.gpu= gpu
        self.labels=num_labels

    def load_data(self, data):

        loaded_data = DataLoader(
            data,  # The training samples.
            sampler = RandomSampler(data), # Select batches randomly
            batch_size = self.config.batch_size # Use this batch size.
            )

        return loaded_data
    
    def create_loaders(self):
        train_set= self.load_data(self.train)
        val_set= self.load_data(self.val)
        test_set= self.load_data(self.test)
        return train_set, val_set, test_set


    def get_model(self):
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = self.labels, # The number of output labels  
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        if self.gpu ==True:
            model.cuda()
        
        return model

    def get_optimizer(self, model):
        optimizer = Adam(model.parameters(),
                  lr = self.config.learning_rate,
                  betas= (0.9,0.98),
                  eps= 1e-06,
                  weight_decay= 0.1) # tabel 9 & 10 RoBERTa paper
                
        return optimizer 

    def get_scheduler(self, train, optimizer):
        epochs = self.config.epochs

        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0.06*total_steps, # linear warmup for the first 6% as in RoBERTa
                                                    num_training_steps = total_steps)
        return scheduler
        
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    #finetune the model etc
    #create evaluation function as well
        





    

 