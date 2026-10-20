import torch

class FWLoss(torch.nn.Module):
    def __init__(self, num_classes, reduction = 'mean', device = 'cpu'):
        super(FWLoss, self).__init__()
        self.reduction = reduction
        kappa = torch.arange(1,num_classes+1).to(device) #  [1, 2,..., # of categories]
        self.kappa_inv = 1 / (kappa)
        self.device = device

    def forward(self, input, target):
        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        z_sorted = torch.sort(z, axis=-1, descending=True).values
        z_sorted_cumsum = z_sorted.cumsum(axis=-1)
        z_hnscs = self.kappa_inv[torch.newaxis,:]*(z_sorted_cumsum - 1)
        # losses = 1 - z_at_y + torch.max(z_hnscs, axis=-1).values
        max_result = torch.max(z_hnscs, dim=-1)

        H = max_result.values
        kappas = max_result.indices + 1   # +1 because indices are 0-based

        losses = 1 - z_at_y + H

        if self.reduction == 'mean':
            return torch.max(z_hnscs, axis=-1).values, losses, losses.mean(), kappas
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class FWLoss_inplace(torch.nn.Module):
    def __init__(self, num_classes, reduction = 'mean', device = 'cpu'):
        super(FWLoss_inplace, self).__init__()
        self.reduction = reduction
        kappa = torch.arange(1,num_classes+1).to(device) #  [1, 2,..., # of categories]
        self.kappa_inv = 1 / (kappa)
        self.device = device

    def forward(self, input, target):
        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        z_hnscs = torch.sort(z, axis=-1, descending=True).values
        z_hnscs.cumsum_(axis=-1)
        z_hnscs.sub_(1)
        z_hnscs.mul_(self.kappa_inv[torch.newaxis,:])
        # z_hnscs = self.kappa_inv[torch.newaxis,:]*(z_sorted - 1)
        losses = 1 - z_at_y + torch.max(z_hnscs, axis=-1).values

        # losses = 1 - z_at_y + torch.max(self.kappa_inv[torch.newaxis,:]*(z_hnscs - 1), axis=-1).values
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class FWLoss_topk(torch.nn.Module):
    def __init__(self, num_classes, reduction = 'mean', device = 'cpu', ktopk_init = 128, growth_rate = 8):
        assert(reduction in ['mean', 'sum'])

        super(FWLoss_topk, self).__init__()
        self.reduction = reduction
        self.ktopk_init = ktopk_init
        kappa = torch.arange(1,num_classes+1).to(device) #  [1, 2,..., # of categories]
        self.kappa_inv = 1 / (kappa)
        self.growth_rate = growth_rate
        self.device = device

    def fallback(self,input,target):
        # this is identical to the naive implementation about
        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        z_sorted = torch.sort(z, axis=-1, descending=True).values
        z_sorted_cumsum = z_sorted.cumsum(axis=-1)
        z_hnscs = self.kappa_inv[torch.newaxis,:]*(z_sorted_cumsum - 1)
        losses = 1 - z_at_y + torch.max(z_hnscs, axis=-1).values
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        
    def forward(self, input, target):
        
        if input.shape[1] <= self.ktopk_init:
            return self.fallback(input,target)
        
        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        ktopk = self.ktopk_init
        loss_val = torch.tensor(0.).to(self.device)

        done = torch.zeros_like(y).to(self.device)

        working_indices = torch.arange(len(y), dtype= torch.long).to(self.device)
        num_classes = z.shape[1]

        while ~torch.all(done) and ktopk < num_classes: # while not all done

            z_sorted = torch.topk(z[working_indices,:], ktopk, axis=-1).values

            z_sorted_cumsum = z_sorted.cumsum(axis=-1)
            z_hnscs = self.kappa_inv[torch.newaxis, :ktopk]*(z_sorted_cumsum - 1)

            z_hnscs_max_vals, z_hnscs_max_idx = torch.max(z_hnscs, axis=-1)

            # if your max takes place within the correct range, then you're all done
            done = z_hnscs_max_idx < ktopk - 1

            # subset the working indices to those that are not all done
            working_indices = working_indices[~done]

            loss_val += z_hnscs_max_vals[done].sum() if torch.any(done) else torch.tensor(0.).to(self.device)

            ktopk *= self.growth_rate

        # fallback for when only a subset
        if ktopk >= num_classes and ~torch.all(done):
            z_sorted = torch.sort(z[working_indices,:], axis=-1, descending=True).values

            z_sorted_cumsum = z_sorted.cumsum(axis=-1)
            z_hnscs = self.kappa_inv[torch.newaxis, :]*(z_sorted_cumsum - 1)

            z_hnscs_max_vals, z_hnscs_max_idx = torch.max(z_hnscs, axis=-1)
            loss_val += torch.max(z_hnscs, axis=-1).values.sum()

        if self.reduction == 'mean':
            loss_val /= len(y)

            loss_val += (1 - z_at_y).mean()
        elif self.reduction == 'sum':
            loss_val += (1 - z_at_y).sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return loss_val





class FWLoss_topk_uniform(torch.nn.Module):
    def __init__(self, num_classes, reduction = 'mean', device = 'cpu', ktopk_init = 128, growth_rate = 8):
        assert(reduction in ['mean', 'sum'])

        super(FWLoss_topk_uniform, self).__init__()
        self.reduction = reduction
        self.ktopk_init = ktopk_init
        kappa = torch.arange(1,num_classes+1).to(device) #  [1, 2,..., # of categories]
        self.kappa_inv = 1 / (kappa)
        self.growth_rate = growth_rate
        self.device = device

    def fallback(self,input,target):
        # this is identical to the naive implementation about
        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        z_sorted = torch.sort(z, axis=-1, descending=True).values
        z_sorted_cumsum = z_sorted.cumsum(axis=-1)
        z_hnscs = self.kappa_inv[torch.newaxis,:]*(z_sorted_cumsum - 1)
        losses = 1 - z_at_y + torch.max(z_hnscs, axis=-1).values
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        
    def forward(self, input, target):
        
        if input.shape[1] <= self.ktopk_init:
            return self.fallback(input,target)
        
        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        ktopk = self.ktopk_init
        loss_val = torch.tensor(0.).to(self.device)

        done = torch.zeros_like(y).to(self.device)

        num_classes = z.shape[1]

        while ktopk < num_classes: # while not all done

            z_sorted = torch.topk(z, ktopk, axis=-1).values

            z_sorted_cumsum = z_sorted.cumsum(axis=-1)
            z_hnscs = self.kappa_inv[torch.newaxis, :ktopk]*(z_sorted_cumsum - 1)

            z_hnscs_max_vals, z_hnscs_max_idx = torch.max(z_hnscs, axis=-1)

            # if your max takes place within the correct range, then you're all done
            done = z_hnscs_max_idx < ktopk - 1

            if torch.all(done):
                loss_val += z_hnscs_max_vals[done].sum()
                break

            ktopk *= self.growth_rate
            
        # if you are here, that means either you are all done
        # or ktopk has exceed ktopk
        
        if ktopk >= num_classes:
            return self.fallback(input,target)
        else:
            
            if self.reduction == 'mean':
                loss_val /= len(y)
    
                loss_val += (1 - z_at_y).mean()
            elif self.reduction == 'sum':
                loss_val += (1 - z_at_y).sum()
            else:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")
    
            return loss_val




class FWLoss_topk_uniform_inplace(torch.nn.Module):
    def __init__(self, num_classes, reduction = 'mean', device = 'cpu', ktopk_init = 128, growth_rate = 8):
        assert(reduction in ['mean', 'sum'])

        super(FWLoss_topk_uniform_inplace, self).__init__()
        self.reduction = reduction
        self.ktopk_init = ktopk_init
        kappa = torch.arange(1,num_classes+1).to(device) #  [1, 2,..., # of categories]
        self.kappa_inv = 1 / (kappa)
        self.growth_rate = growth_rate
        self.device = device

    def fallback(self,input,target):
        # this is identical to the naive implementation about
        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        z_hnscs = torch.sort(z, axis=-1, descending=True).values
        z_hnscs.cumsum_(axis=-1)
        z_hnscs.sub_(1)
        z_hnscs.mul_(self.kappa_inv[torch.newaxis,:])
        # z_hnscs = self.kappa_inv[torch.newaxis,:]*(z_sorted - 1)
        losses = 1 - z_at_y + torch.max(z_hnscs, axis=-1).values

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

    def forward(self, input, target):

        if input.shape[1] <= self.ktopk_init:
            return self.fallback(input,target)

        z = input # has shape (batch_size, num_classes)
        y = target # has shape (batch_size, 1)

        z_at_y = z[torch.arange(z.shape[0]).to(self.device), y]

        ktopk = self.ktopk_init
        loss_val = torch.tensor(0.).to(self.device)

        done = torch.zeros_like(y).to(self.device)

        num_classes = z.shape[1]

        while ktopk < num_classes: # while not all done

            # this is impossible to make in place
            z_hnscs = torch.topk(z, ktopk, axis=-1).values

            z_hnscs.cumsum_(axis=-1)
            z_hnscs.sub_(1)
            z_hnscs.mul_(self.kappa_inv[torch.newaxis,:ktopk])

            z_hnscs_max_vals, z_hnscs_max_idx = torch.max(z_hnscs, axis=-1)

            # if your max takes place within the correct range, then you're all done
            done = z_hnscs_max_idx < ktopk - 1

            if torch.all(done):
                loss_val += z_hnscs_max_vals[done].sum()
                break

            ktopk *= self.growth_rate

        # if you are here, that means either you are all done
        # or ktopk has exceed ktopk

        if ktopk >= num_classes:
            return self.fallback(input,target)
        else:

            if self.reduction == 'mean':
                loss_val /= len(y)

                loss_val += (1 - z_at_y).mean()
            elif self.reduction == 'sum':
                loss_val += (1 - z_at_y).sum()
            else:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")

            return loss_val
