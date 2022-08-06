from baseline import Baseline
from maml import MAML
from model import Model
from sinusoid import SinusoidDataset

# Dataset hyperparameters
num_outer_steps = 15000
batch_size = 25
num_train_tasks = batch_size * num_outer_steps
num_shots_support = 10
num_shots_query = 10
num_test_tasks = 1000

# Model hyperparameters
num_inner_steps = 1
inner_lr = 1e-2
outer_lr = 1e-3
log_interval = 250
num_inner_steps_test = 10
baseline_lr = 0.01

# Collect data
dataset = SinusoidDataset()
data_train = dataset.get_data(batch_size, num_shots_support,
                              num_shots_query, num_train_tasks)
data_test = dataset.get_data(1, num_shots_support, num_shots_query,
                             num_test_tasks)

# Initialize MAML, FOMAML, and baseline
maml = MAML(Model(), num_inner_steps, inner_lr, outer_lr, False)
fomaml = MAML(Model(), num_inner_steps, inner_lr, outer_lr, True)
baseline = Baseline(Model(), num_inner_steps, baseline_lr)

# Train and test models
models = [maml, fomaml, baseline]
names = ["MAML", "FOMAML", "baseline"]
for model, name in zip(models, names):
    model.train(data_train, log_interval, f"./logs/{name}_train.json")
    model.test(data_test, num_inner_steps_test)