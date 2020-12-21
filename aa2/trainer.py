
import os
import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

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


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparameters, model_name):
        # epoch = epoch
        # model =  a train pytroch model
        # optimizer = a pytorch Optimizer
        # loss = loss (detach it from GPU)
        # scores = dict where keys are names of metrics and values the value for the metric
        # hyperparameters = dict of hyperparameters
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
                        'hyperparameters': hyperparameters,
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
        hyperparameters = chp['hyperparameters']
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
        print(loss_function)
        mod_name = hyperparameters['model_name']
        hid_state = hyperparameters['hid_state'] # hid state test
        batch_size = hyperparameters['batch_size']
        
        device = self.device
        
        
        batcher = Batcher(train_X, train_y, device, batch_size=batch_size, max_iter=epochs) # epochs of batches 
        in_size = train_X.shape[2] # = 5 : num_features
        out_size = 102
        if loss_function == "CrossEntropyLoss":
            #weights = torch.tensor([1.0, 29.24, 50.71, 181.33, 31.36]).to(device) # weights test
            #loss = getattr(nn, loss_function)(weight=weights)# weights test
            if hid_state:
                raise Exception('CrossEntropyLoss and last hidden state combination not allowed')
            else:
                out_size = 5
        else:
           # loss = getattr(nn, loss_function) # weights test
            if hid_state:
                out_size = 102 # hid state test
            else:
                out_size = 1
            
            
        m = model_class(in_size, hid_size, out_size, n_layers, hid_state) # hid state test
        m = m.to(device)
        
        loss = getattr(nn, loss_function)
        opt = getattr(optim, izer)(m.parameters(), lr=lr)
        
        print(f"Model: {mod_name}")
        epoch = 0
        for e in batcher:
            m.train()
            tot_loss = 0
            for batch, labels in e:
                opt.zero_grad()
                #print("MODEL:", m)
                #print("BATCH SHAPE", batch.shape)
                out = m(batch.float().to(device), device)                
                #print("OUT SHAPE", out.shape)
                #print("LABELS SHAPE", labels.shape)
                
                if loss_function == "CrossEntropyLoss": # cross ent wants the out in a diff shape compared to mse and mae
                    if not hid_state:
                        out = out.reshape(out.shape[0], out.shape[2], out.shape[1])
                        labels = labels.long()
                    else: # hid state test
                        out = out.unsqueeze(1)
                        labels = labels.long()
                    #print("OUT SHAPE", out.shape)
                    #print("LABELS SHAPE", labels.shape)
                    #l = loss(out.to(device), labels.to(device)) # weights test
                else:
                    if not hid_state: # hid state test
                        out = out.squeeze(2)
                        labels = labels.float()
                l = loss()(out.to(device), labels.to(device))
                tot_loss += l
                l.backward()
                opt.step()
            print("Total loss in epoch {} is {}".format(epoch, tot_loss))        
            epoch += 1
            
            
#           validation
            pred = []
            y = []
            m.eval()
            val_batcher = Batcher(val_X, val_y, device, batch_size=batch_size, max_iter=1)
            for val_e in val_batcher:
                for batch, labels in val_e:
                    with torch.no_grad():
                        out = m(batch.float().to(device), device)
                        if loss_function == "CrossEntropyLoss":
                            _, out = torch.max(out, 2)
                        else:
                            out = torch.round(out)
                        
                        out = out.reshape(-1).tolist()                        
                        labels = labels.reshape(-1).tolist()
                pred.extend(out)
                y.extend(labels)
        
        acc = accuracy_score(y, pred, normalize=True)
        prec = precision_score(y, pred, average='weighted')
        rec = recall_score(y, pred, average='weighted')
        f1 = f1_score(y, pred, average='weighted')
        
        scores = {
            "accuracy" : acc,
            "precision": prec,
            "recall"   : rec,
            "f1-score" : f1
        }
        
        print(f"Model name: {mod_name}, \nAccuracy: {acc}, \nPrecision: {prec}, \nRecall: {rec}, \nF1-Score: {f1}")
        
        tot_loss = tot_loss.to('cpu')
        
        self.save_model(e, m, opt, tot_loss, scores, hyperparameters, mod_name)
        
        pass


    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        epoch, model_state_dict, optimizer_state_dict, hyperparameters, loss, scores, model_name = load_model(best_model_path)
        
        in_size = test_X.shape[2]
        n_layers = hyperparameters['number_layers']
        hid_size = hyperparameters['hid_dim']
        loss_function = hyperparameters['loss_funct']
        device = self.device
        if loss_function == "CrossEntropyLoss":
            if hid_state:
                raise Exception('CrossEntropyLoss and last hidden state combination not allowed')
            else:
                out_size = 5
        else:
            if hid_state:
                out_size = 102 # hid state test
            else:
                out_size = 1
        
        
        model = model_class(in_size, hid_size, out_size, n_layers, hid_state)
        model.load_state_dict(torch.load(model_state_dict))
        m = m.to(device)
        model.eval()
        pred = []
        y = []
        m.eval()
        val_batcher = Batcher(val_X, val_y, device, batch_size=batch_size, max_iter=1)
        for val_e in val_batcher:
            for batch, labels in val_e:
                with torch.no_grad():
                    out = m(batch.float().to(device), device)
                    if loss_function == "CrossEntropyLoss":
                        _, out = torch.max(out, 2)
                    else:
                        out = torch.round(out)

                    out = out.reshape(-1).tolist()                        
                    labels = labels.reshape(-1).tolist()
                pred.extend(out)
                y.extend(labels)
        
        acc = accuracy_score(y, pred, normalize=True)
        prec = precision_score(y, pred, average='weighted')
        rec = recall_score(y, pred, average='weighted')
        f1 = f1_score(y, pred, average='weighted')
        
        scores = {
            "accuracy" : acc,
            "precision": prec,
            "recall"   : rec,
            "f1-score" : f1
        }
        
        print(f"Model name: {mod_name}, \nAccuracy: {acc}, \nPrecision: {prec}, \nRecall: {rec}, \nF1-Score: {f1}")

        
        
        pass
