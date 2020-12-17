
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Batcher:
    def __init__(self, X, y, device, batch_size=50, max_iter=None):
        self.X = X
        self.y = y
        self.device = device
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.curr_iter = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_iter == self.max_iter:
            raise StopIteration
        permutation = torch.randperm(self.X.shape[0], device=self.device)
        permX = self.X[permutation]
        permy = self.y[permutation]
        splitX = torch.split(permX, self.batch_size)
        splity = torch.split(permy, self.batch_size)
        
        self.curr_iter += 1
        
        return zip(splitX, splity)
        
class Trainer:


    def __init__(self, dump_folder="/tmp/aa2_models/", device=torch.device("cuda:3")):
        self.dump_folder = dump_folder
        self.device = device
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparamaters = dict of hyperparamaters
        # model_name = name of the model you have trained, make this name unique for each hyperparamater.  I suggest you name them:
        # model_1, model_2 etc 
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self, path):
        # Finish this function so that it loads a model and return the appropriate variables
        chp = torch.load(path)
        epoch = chp['epoch']
        model_state_dict = chp['model_state_dict']
        optimizer_state_dict = chp['optimizer_state_dict']
        hyperparameters = chp['hyperparamaters']
        loss = chp['loss']
        scores = chp['scores']
        model_name = chp['model_name']
        
        return epoch, model_state_dict, optimizer_state_dict, hyperparameters, loss, scores, model_name


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparameters):
        # Finish this function so that it set up model then trains and saves it.
        
        
        lr = hyperparameters['learning_rate']
        n_layers = hyperparameters['number_layers']
        epochs = hyperparameters['epochs']
        hid_size = hyperparameters['hid_dim']
        izer = hyperparameters['optimizer']
        loss_function = hyperparameters['loss_funct']
        mod_name = hyperparameters['model_name']
        
        device = self.device
        
        
        batches = Batcher(train_X, train_y, device, max_iter=epochs)
        
        in_size = train_X.shape[2] # = 5 : num_features
        out_size = 5 # jo nu är den det igen # nej nu är det något annat välkommen # num_labels, len of id2ner
        
        m = model_class(in_size, hid_size, out_size, n_layers)
        m = m.to(device)
        loss = getattr(nn, loss_function)
        opt = getattr(optim, izer)(m.parameters(), lr=lr)
        
        epoch = 0
        for batch in batches:
            tot_loss = 0
            for sent, labels in batch:
                opt.zero_grad()
                print("MODEL:", m)
                print("SENT SHAPE", sent.shape)
                out = m(sent.to(device), device)
                print("OUT SHAPE", out.shape)
                print("LABELS SHAPE", labels.shape)
                print("LABELS", labels)
                out = out.reshape(out.shape[0]*out.shape[1], out.shape[2])
                print("NEW OUT SHAPE", out.shape)
                labels = labels.reshape(labels.shape[0]*labels.shape[1], 1)
                #out = out.reshape(out.shape[0], out.shape[2], out.shape[1]) #gör den till [50, 5, 102] ist för [50, 102, 5]
                l = loss()(out.to(device), labels.to(device))
                tot_loss += l
                l.backward()
                opt.step()
                aosjdnoais
            print("Total loss in epoch {} is {}".format(epoch, tot_loss))
            epoch += 1
        print('hello got all the way here')
                
        
        
        #loss crossent, l1
        
        
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        pass
