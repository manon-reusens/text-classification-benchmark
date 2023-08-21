import sys
import os
import wandb
os.environ['WANDB_DIR']= '/lustre1/scratch/344/vsc34470/wandb'
import pandas as pd 
import torch
import random
import numpy as np
import time
sys.path.append(os.getcwd())

import wandb
import pandas as pd 
from nltk import download
download('stopwords',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/roberta/nltk_data')
download('omw-1.4',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/roberta/nltk_data')
download('punkt',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/roberta/nltk_data')
download('wordnet',quiet=True,download_dir='/data/leuven/344/vsc34470/miniconda3/envs/roberta/nltk_data');

from codecarbon import EmissionsTracker
from util.dataloader import DataLoader
from util.datasplitter import data_splitter
#from evaluator import evaluate_classifier
from py_files.prep_data_RoBERTa_ import prep_data_RoBERTa
from py_files.RoBERTa import RoBERTa_model
import torch
from nltk import download
import warnings
warnings.filterwarnings('ignore')
import yaml
from torchmetrics import F1Score
os.environ["WANDB_SILENT"] = "true"




# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
    gpu=True

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    gpu=False



SEED=int(sys.argv[1])
seed_val = SEED
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

if gpu==True:
    torch.cuda.manual_seed_all(seed_val)

path= '/lustre1/scratch/344/vsc34470/results/roberta/'

dl = DataLoader(['emotion'])
data = dl.load()


train_emo, val_emo, test_emo = data_splitter(data['MELD'],0,create_val_set=True,seed=SEED)

data_prep=prep_data_RoBERTa(train_emo, val_emo, test_emo)
train_emo, val_emo, test_emo, num_labels= data_prep.prepare_dataset()

#Load the template yaml sweep config file for BERT
with open("config/roberta_sweep.yaml", 'r') as stream:
    sweep_config = yaml.safe_load(stream)

name = 'emotion_meld_roberta' #change here
sweep_config['name'] =  name
sweep_id = wandb.sweep(sweep_config, project="roberta")



iterations=6

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss,model,sweep_id, path):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config,project='roberta', group=name) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        torch.cuda.empty_cache()
        path= '/lustre1/scratch/344/vsc34470/results/roberta/'+sweep_id+'-model-'+run.name
        config = wandb.config
        roberta=RoBERTa_model(device, config, train_emo, val_emo,test_emo, num_labels, gpu)
        train, val, test = roberta.create_loaders()
        model=roberta.get_model()
        model.to(device)
        optimizer=roberta.get_optimizer(model)
        scheduler=roberta.get_scheduler(train, optimizer)

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        early_stopper = EarlyStopper(patience=0, min_delta=0)


        # For each epoch...
        for epoch_i in range(0, config.epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            ep=epoch_i

            # Reset the total loss for this epoch.
            total_train_loss = 0

            model.train() #put model in training mode

            # For each batch of training data...
            for step, batch in enumerate(train):
              # Progress update every 40 batches.
              if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                
                elapsed = roberta.format_time(time.time() - t0)
                print(elapsed)
                    
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train), elapsed))

              # Unpack this training batch from our dataloader. 
              #
              # As we unpack the batch, we'll also copy each tensor to the GPU using the 
              # `to` method.
              #
              # `batch` contains three pytorch tensors:
              #   [0]: input ids 
              #   [1]: attention masks
              #   [2]: labels 
              b_input_ids = batch[0].to(device)
              b_input_mask = batch[1].to(device)
              b_labels = batch[2].to(device)

              # Always clear any previously calculated gradients before performing a
              # backward pass. PyTorch doesn't do this automatically because 
              # accumulating the gradients is "convenient while training RNNs". 
              # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
              model.zero_grad()        

              # Perform a forward pass (evaluate the model on this training batch).
              # The documentation for this `model` function is here: 
              # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
              # It returns different numbers of parameters depending on what arguments
              # arge given and what flags are set. For our useage here, it returns
              # the loss (because we provided labels) and the "logits"--the model
              # outputs prior to activation.
              loss, logits = model(b_input_ids, 
                                  token_type_ids=None, 
                                  attention_mask=b_input_mask, 
                                  labels=b_labels, return_dict=False)
              
              wandb.log({'train_batch_loss':loss.item()})
              # Accumulate the training loss over all of the batches so that we can
              # calculate the average loss at the end. `loss` is a Tensor containing a
              # single value; the `.item()` function just returns the Python value 
              # from the tensor.
              total_train_loss += loss.item()

              # Perform a backward pass to calculate the gradients.
              loss.backward()

              # Clip the norm of the gradients to 1.0.
              # This is to help prevent the "exploding gradients" problem.
              #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

              # Update parameters and take a step using the computed gradient.
              # The optimizer dictates the "update rule"--how the parameters are
              # modified based on their gradients, the learning rate, etc.
              optimizer.step()

              # Update the learning rate.
              scheduler.step()
            print('Now we will calculate the average loss')

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train)   
            print(avg_train_loss)         
            
            # Measure how long this epoch took.
            training_time = roberta.format_time(time.time() - t0)
            wandb.log({'avg_train_loss':avg_train_loss})
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_f1macro=0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in val:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        

                   (loss, logits) = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        return_dict=False)
                    
                # Accumulate the validation loss.
                total_eval_loss += loss.item()
                
                # Move logits and labels to CPU
                logits = logits.cpu().detach().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += roberta.flat_accuracy(logits, label_ids)

                preds = np.argmax(logits, axis=1)
                tensor_preds= torch.from_numpy(preds).to(device)
                
                #calculate the macro f1-measure
                f1 = F1Score(num_classes=num_labels, average='macro').to(device)
                total_f1macro += f1(tensor_preds, b_labels)
                

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(val)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(val)
            avg_f1macro= total_f1macro/len(val)
            
            # Measure how long the validation run took.
            validation_time = roberta.format_time(time.time() - t0)
           
            wandb.log({'epoch':ep,'SEED':SEED,'val_accuracy':avg_val_accuracy,'avg_val_loss':avg_val_loss, 'f1macro':avg_f1macro}) 
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            if early_stopper.early_stop(avg_val_loss, model, sweep_id, path):   
              print('we stop here and save the model')          
              break

        #add artifact that saves everything
        #check that when using multiple runs, you reinitiate the model every time.

sweep_id = wandb.sweep(sweep_config, project="bayes_roberta")

#Track emissions
tracker = EmissionsTracker(project_name=name,log_level='warning', measure_power_secs=300,
                           output_file='output/emissions_lstm.csv')
#Launch the agent
tracker.start()
wandb.agent(sweep_id, function= train, count=iterations)
tracker.stop()


