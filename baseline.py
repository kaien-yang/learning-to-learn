import json
import os

import numpy as np
import torch
from torch import nn

class Baseline():
    """The baseline model is jointly pre-trained on all meta-training tasks
    and, at test-time, is fine-tuned with a small number of optimization steps
    to new tasks. The structure for this class is analagous to that of the
    MAML class.
    """
    def __init__(self, model, num_inner_steps, lr):
        self.model = model
        self.num_inner_steps = num_inner_steps
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.loss_function = nn.MSELoss()

    def inner_loop(self, x_support, y_support, train):
        support_losses = []
        if train:
            model = self.model
            optimizer = self.optimizer
        else:
            model = type(self.model)()
            model.load_state_dict(self.model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), self.lr)
        for inner_step in range(self.num_inner_steps+1):
            pred_y_support = model.forward(x_support)
            support_loss = self.loss_function(pred_y_support, y_support)
            support_losses.append(support_loss.item())
            optimizer.zero_grad()
            support_loss.backward()
            if inner_step == 0:
                init_grad = torch.concat(
                    [w.grad.flatten() for w in model.parameters()]
                )
                init_grad_norm = torch.linalg.norm(init_grad).item()
            if inner_step == self.num_inner_steps:
                break
            optimizer.step()
        return support_losses, init_grad_norm

    def inner_loop_plot(self, x_support, y_support, x, inner_steps_plot):
        pred_y = {}
        model = type(self.model)()
        model.load_state_dict(self.model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), self.lr)
        for inner_step in range(max(inner_steps_plot)+1):
            pred_y_support = model.forward(x_support)
            support_loss = self.loss_function(pred_y_support, y_support)
            if inner_step == 0 or inner_step in inner_steps_plot:
                pred_y[inner_step] = model.forward(x).detach().cpu().numpy()
            model.zero_grad()
            support_loss.backward()
            optimizer.step()
        return pred_y
    
    def outer_step(self, task_batch, train):
        query_loss_batch = []
        support_losses_batch = []
        init_grad_norm_batch = []
        
        for task in task_batch:
            x_support, y_support, x_query, y_query = task
            support_losses, init_grad_norm = self.inner_loop(x_support, y_support, train)
            support_losses_batch.append(support_losses)
            init_grad_norm_batch.append(init_grad_norm)
            pred_y_query = self.model.forward(x_query)
            query_loss_batch.append(self.loss_function(pred_y_query, y_query))
        
        mean_query_loss = torch.stack(query_loss_batch).mean()
        mean_support_losses = np.mean(np.array(support_losses_batch), axis=0)
        mean_init_grad_norm = np.mean(np.array(init_grad_norm_batch))
        return mean_query_loss, mean_support_losses, mean_init_grad_norm
    
    def train(self, data_train, log_interval, filename="./logs/train.json"):
        log = {}
        query_losses = []
        init_grad_norms = []
        for outer_step, task_batch in enumerate(data_train):
            mean_query_loss, mean_support_losses, mean_init_grad_norm = \
                self.outer_step(task_batch, train=True)
            if outer_step % log_interval == 0:
                print(f"Iteration {outer_step}: ")
                print("Baseline loss (query set loss, batch average): ",
                      f"{mean_query_loss.item():.3f}")
                print("Pre-adaptation support set loss (batch average): ",
                      f"{mean_support_losses[0]:.3f}")
                print("Post-adaptation support set loss (batch average): ",
                      f"{mean_support_losses[-1]:.3f}")
                print("Norm of initial adaptation gradient (batch average): ",
                      f"{mean_init_grad_norm:.3f}")
                print("-"*50)
                log[outer_step] = {
                    "Query set loss": mean_query_loss.item(),
                    "Pre-adaptation support set loss": mean_support_losses[0],
                    "Post-adaptation support set loss": mean_support_losses[-1],
                    "Norm of initial adaptation gradient": mean_init_grad_norm,
                }
                query_losses.append(mean_query_loss.item())
                init_grad_norms.append(mean_init_grad_norm)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(log, f)
        return query_losses, init_grad_norms
    
    def test(self, data_test, num_inner_steps=None):
        query_losses = []
        support_losses = []
        num_test_tasks = len(data_test) * len(data_test[0])
        if num_inner_steps is not None:
            prev_num_inner_steps = self.num_inner_steps
            self.num_inner_steps = num_inner_steps
        for task_batch in data_test:
            q_loss, s_losses, _ = self.outer_step(task_batch, train=False)
            query_losses.append(q_loss.item())
            support_losses.append(s_losses)
        if num_inner_steps is not None:
            self.num_inner_steps = prev_num_inner_steps
        mean_query_loss = np.mean(query_losses)
        CI_95_query_loss = 1.96 * np.std(query_losses) / np.sqrt(num_test_tasks)
        np_support_losses = np.array(support_losses)
        mean_support_losses = np.mean(np_support_losses, axis=0)
        CI_95_support_losses = (1.96 * np.std(np_support_losses, axis=0)
                                / np.sqrt(num_test_tasks))
        print(f"Evaluation statistics on {num_test_tasks} test tasks: ")
        print("Baseline loss:")
        print(f"Mean: {mean_query_loss:.3f}")
        print(f"95% confidence interval: {CI_95_query_loss:.3f}")
        return (mean_query_loss, CI_95_query_loss,
                mean_support_losses, CI_95_support_losses)