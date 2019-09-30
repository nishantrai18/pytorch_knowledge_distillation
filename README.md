# Task
Perform training of ConvNets using **Knowledge Distillation** and **Swish** activation function on the **CIFAR-100** dataset.

# Instructions
The scripts `installDeps.sh` and `runSanityCheck.sh` are responsible for ensuring that the build and code version are in a valid state

- `installDeps.sh`: Installs **dependencies** required to run the main program
- `runSanityCheck.sh`: Runs **unit tests** for all the components, and ensures that the code is in a valid state

Use the following commands to get started,

	> bash installDeps.sh
	> bash runSanityCheck.sh

# Running the code

We discuss the code structure in this section. The main functionalities are present in the following files,

- `main.py`: Driver program to perform different tasks. Use --help for details.
- `kd_module.py`: Contains functionality and implementation of Knowledge distillation models
- `training_utils.py`: Contains utilities useful for training using different schemes
- `model_utils.py`: Contains implementation of model training class
- `swish/`: Contains swish module related components
- `models/`: Contains definitions of various DCNNs

We also have helper utility and classes to supplement the driver scripts

- `args.py`: Contains arguments for different scripts
- `cached_dataset.py`: Defines the cached dataset reader and generator
- `metric_utils.py`: Defines helper class to perform metrics tracking and logging
- `dataset_utils.py`: Helper class for different dataset dataloaders

Use the following commands to get started with training,

```
// Train base net in a straight-forward way with Swish activation
> python main.py --task base_tr --base-model basenet --activation swish 
    --perform-data-aug True --notes 30sep --epochs 30
// Train resnet18 in a straight-forward way with relu activation
> python main.py --task base_tr --base-model resnet18 --activation relu
    --perform-data-aug True --notes 30sep --epochs 30
// Perform on the fly knowledge distillation with simple basenet studet model and a provided teacher checkpoint
> python main.py  --task kd_otf --student-model basenet --activation relu
    --teacher-ckpt-pth ../model_ckpt/resnet18_init/cifar100_9.pt
    --temperature 1.0 --kd-weight 0.5
    --perform-data-aug True --notes 30sep --epochs 30
// Perform cached knowledge distillation with an ensemble of teachers
> python main.py  --task kd_cached --student-model basenet --activation relu
    --teachers resnet18_sqnet_mobnet2
    --temperature 1.0 --kd-weight 0.5
    --perform-data-aug True --notes 30sep --epochs 30
```

Use the following commands to generate new cached datasets,

	> python cached_dataset.py

# Other notes

**swish/** contains pre-compiled binaries in order to allow out-of-the-box execution
