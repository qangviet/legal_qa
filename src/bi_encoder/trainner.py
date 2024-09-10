import torch
from torch.utils.checkpoint import get_device_states, set_device_states
import time
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from .model import BiEncoder
from .loss import BiEncoderNllLoss
import matplotlib.pyplot as plt

class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None
             
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

class BiEncoderTrainer():
    
    def __init__(self,
                args,
                train_dataloader,
                val_dataloader):
        self.parallel = True if torch.cuda.device_count() > 1 else False
        print("No of GPU(s):",torch.cuda.device_count())
        self.args = args
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiEncoder(model_checkpoint=self.args.BE_checkpoint,
                              representation=self.args.BE_representation,
                              freeze=self.args.BE_freeze)
        
        if self.args.path_weight_mlm is not None:
            self.model.encoder.copy_weight_from_MLM_model(self.args.path_weight_mlm)
            print("Copy weight MLM model:", self.args.path_weight_mlm)
        
        if self.parallel:
            print("Parralel Training")
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        self.loss_fn = BiEncoderNllLoss(score_type=self.args.BE_score)
        self.optimizer = AdamW(self.model.parameters(), lr=args.BE_lr)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 0.1 * len(self.train_dataloader) * self.args.BE_num_epochs, 
                                              len(self.train_dataloader) * self.args.BE_num_epochs)
        self.epoch=0
        self.patience_counter = 0
        self.best_val_acc = 0.0
        self.epochs_count = []
        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []
        
    def train_biencoder(self):
        print("\n",
              20 * "=",
              "Validation before training",
              20 * "=")
        val_time, val_loss, val_acc = self.validate()
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(val_time, val_loss, (val_acc*100)))
        print("\n",
              20 * "=",
              "Training bi-encoder model on device: {}".format(self.device),
              20 * "=")
        while self.epoch < self.args.BE_num_epochs:
            self.epoch += 1
            self.epochs_count.append(self.epoch)
            print("* Training epoch {} / {}:".format(self.epoch, self.args.BE_num_epochs))
            epoch_train_time, epoch_train_loss, epoch_train_acc = self.train()
            self.train_losses.append(epoch_train_loss)
            self.train_acc.append(epoch_train_acc.to('cpu') * 100)
            print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                  .format(epoch_train_time, epoch_train_loss, (epoch_train_acc*100)))
            
            print("* Validation for epoch {} / {}:".format(self.epoch, self.args.BE_num_epochs))
            epoch_val_time, epoch_val_loss, epoch_val_acc = self.validate()
            self.valid_losses.append(epoch_val_loss)
            self.valid_acc.append(epoch_val_acc.to('cpu')*100)
            print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
                  .format(epoch_val_time, epoch_val_loss, (epoch_val_acc*100)))
            
            if epoch_val_acc < self.best_val_acc:
                self.patience_counter += 1
            
            else:
                self.best_val_acc = epoch_val_acc
                self.patience_counter = 0
                if self.parallel:
                    self.model.module.encoder.save(self.args.biencoder_path)
                
                else:
                    self.model.encoder.save(self.args.biencoder_path)
                print(f"Save encoder to (best acc:{self.best_val_acc}):", self.args.biencoder_path)
                
            if self.epoch == self.args.BE_num_epochs:
                if self.parallel:
                    self.model.module.encoder.save(self.args.final_path)
                else:
                    self.model.encoder.save(self.args.final_path)
            
            if self.parallel:
                checkpoint = { 
                    'epoch': self.epoch,
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
                torch.save(checkpoint, 'last_checkpoint.pth')
            else:
                checkpoint = { 
                    'epoch': self.epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()}
                torch.save(checkpoint, 'last_checkpoint.pth')
        
        # Plotting of the loss curves for the train and validation sets.
        plt.figure()
        plt.plot(self.epochs_count, self.train_losses, "-r")
        plt.plot(self.epochs_count, self.valid_losses, "-b")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["Training loss", "Validation loss"])
        plt.title("Cross entropy loss")
        plt.show()
    
        plt.figure()
        plt.plot(self.epochs_count, self.train_acc, '-r')
        plt.plot(self.epochs_count, self.valid_acc, "-b")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(["Training accuracy", "Validation accuracy"])
        plt.title("Accuracy")
        plt.show()
    
        #return the final q_model, ctx_model
        if self.parallel:
            return self.model.module.get_model()
        else:
            return self.model.get_model()
    
    def train(self):
        self.model.train()
        epoch_start = time.time()
        batch_time_avg = 0.0
        epoch_loss = 0.0
        epoch_correct = 0
        tqdm_batch_iterator = tqdm(self.train_dataloader)
        for i, batch in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            if self.args.grad_cache:
                loss, num_correct = self.step_cache(batch)
            else:
                loss, num_correct = self.step(batch)
    
            batch_time_avg += time.time() - batch_start
            epoch_loss += loss
            epoch_correct += num_correct

            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                        .format(batch_time_avg/(i+1),
                        epoch_loss/(i+1))
            tqdm_batch_iterator.set_description(description)

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = epoch_loss / len(self.train_dataloader)
        epoch_accuracy = epoch_correct / len(self.train_dataloader.dataset)

        return epoch_time, epoch_avg_loss, epoch_accuracy, 
                
    def step(self, batch):
        if self.args.no_hard != 0:
            q_input_ids, q_attn_mask, p_input_ids, p_attn_mask, n_input_ids, n_attn_mask = tuple(t.to(self.device) for t in batch)
            doc_len = p_input_ids.shape[-1]
            n_input_ids = n_input_ids.view(-1, doc_len)
            n_attn_mask = n_attn_mask.view(-1, doc_len)
            doc_input_ids = torch.cat((p_input_ids, n_input_ids), 0)
            doc_attn_mask = torch.cat((p_attn_mask, n_attn_mask), 0)
            
        else:
            q_input_ids, q_attn_mask, doc_input_ids, doc_attn_mask = tuple(t.to(self.device) for t in batch)
        
        self.optimizer.zero_grad()
        
        q_vectors, doc_vectors = self.model(q_input_ids, q_attn_mask, doc_input_ids, doc_attn_mask)
        loss, num_correct = self.loss_fn.calc(q_vectors, doc_vectors)
        loss.backward()
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item(), num_correct
    
    def step_cache(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        if self.args.no_hard != 0:
            q_input_ids, q_attn_mask, p_input_ids, p_attn_mask, n_input_ids, n_attn_mask = tuple(t.to(self.device) for t in batch)
            doc_len = p_input_ids.shape[-1]
            n_input_ids = n_input_ids.view(-1, doc_len)
            n_attn_mask = n_attn_mask.view(-1, doc_len)
            doc_input_ids = torch.cat((p_input_ids, n_input_ids), 0)
            doc_attn_mask = torch.cat((p_attn_mask, n_attn_mask), 0)
            
        else:
            q_input_ids, q_attn_mask, doc_input_ids, doc_attn_mask = tuple(t.to(self.device) for t in batch)
            
        all_q_reps, all_doc_reps = [], []
        q_rnds, doc_rnds = [], []
        
        q_id_chunks = q_input_ids.split(self.args.q_chunk_size)
        q_attn_mask_chunks = q_attn_mask.split(self.args.q_chunk_size)
        
        doc_id_chunks = doc_input_ids.split(self.args.doc_chunk_size)
        doc_attn_mask_chunks = doc_attn_mask.split(self.args.doc_chunk_size)
        
        for id_chunk, attn_chunk in zip(q_id_chunks, q_attn_mask_chunks):
            q_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                q_chunk_reps = self.model(id_chunk, attn_chunk, None, None)[0]
            all_q_reps.append(q_chunk_reps)
        all_q_reps = torch.cat(all_q_reps)
        
        for id_chunk, attn_chunk in zip(doc_id_chunks, doc_attn_mask_chunks):
            doc_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                doc_chunk_reps = self.model(None, None, id_chunk, attn_chunk)[1]
            all_doc_reps.append(doc_chunk_reps)
        all_doc_reps = torch.cat(all_doc_reps)
        
        all_q_reps = all_q_reps.float().detach().requires_grad_()
        all_doc_reps = all_doc_reps.float().detach().requires_grad_()
        loss, num_correct = self.loss_fn.calc(all_q_reps, all_doc_reps)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        
        q_grads = all_q_reps.grad.split(self.args.q_chunk_size)
        doc_grads = all_doc_reps.grad.split(self.args.doc_chunk_size)
        
        for id_chunk, attn_chunk, grad, rnd in zip(q_id_chunks, q_attn_mask_chunks, q_grads, q_rnds):
            with rnd:
                q_chunk_reps = self.model(id_chunk, attn_chunk, None, None)[0]
                surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())
        
            surrogate.backward()

                
        for id_chunk, attn_chunk, grad, rnd in zip(doc_id_chunks, doc_attn_mask_chunks, doc_grads, doc_rnds):
            with rnd:
                doc_chunk_reps = self.model(None, None, id_chunk, attn_chunk)[1]
                surrogate = torch.dot(doc_chunk_reps.flatten().float(), grad.flatten())
                
            surrogate.backward()
            
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item(), num_correct
    
    def validate(self):
        self.model.eval()
        
        epoch_start = time.time()
        
        total_loss = 0.0
        total_correct = 0
        accuracy = 0
        
        with torch.no_grad():
            tqdm_batch_iterator = tqdm(self.val_dataloader)
            for i, batch in enumerate(tqdm_batch_iterator):
                if self.args.no_hard != 0:
                    q_input_ids, q_attn_mask, p_input_ids, p_attn_mask, n_input_ids, n_attn_mask = tuple(t.to(self.device) for t in batch)
                    doc_len = n_input_ids.size()[-1]
                    n_input_ids = n_input_ids.view(-1,doc_len)
                    n_attn_mask = n_attn_mask.view(-1,doc_len)
                    doc_input_ids = torch.cat((p_input_ids, n_input_ids), 0)
                    doc_attn_mask = torch.cat((p_attn_mask, n_attn_mask), 0)
                else:
                    q_input_ids, q_attn_mask, doc_input_ids, doc_attn_mask = tuple(t.to(self.device) for t in batch)

                q_vectors, doc_vectors = self.model(q_input_ids, q_attn_mask, doc_input_ids, doc_attn_mask)
                loss, num_correct = self.loss_fn.calc(q_vectors, doc_vectors)
                total_loss += loss.item()
                total_correct += num_correct

            epoch_time = time.time() - epoch_start
            val_loss = total_loss / len(self.val_dataloader)
            accuracy = total_correct / len(self.val_dataloader.dataset)

        return epoch_time, val_loss, accuracy