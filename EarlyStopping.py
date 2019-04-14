class EarlyStopper():
    def __init__(self, patience=8):
        self.patience = patience
        self.stop = False
        self.val_loss = 99999999
        self.counter = 0

    def check_loss_history(self, loss):
        if loss < self.val_loss:
            self.val_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            print("Early stopping counter {0} out of {1}".format(self.counter, self.patience))
            if self.counter == self.patience:
                self.stop = True


    
