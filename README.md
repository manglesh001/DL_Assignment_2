# DL_Assignment_2
##  Train a CNN from Scratch & Hyperparameter Tuning:
##Part A
This project involves training a Convolutional Neural Network (CNN) from scratch on the iNaturalist dataset. The goal is to explore the complete model development lifecycle—starting from designing the architecture to tuning hyperparameters like learning rate, batch size, dropout, and filter configurations.

for Q1  Just Create CNN class using 5 layers 

## train using hyperparameter tuning 
# wandb arguments
    parser.add_argument("--wandb_entity", default="manglesh-patidar-cs23m025")
    parser.add_argument("--wandb_project", default="inaturalist-cnn")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--num_filters", type=int, nargs=5, default=[32]*5)
    parser.add_argument("--filter_sizes", type=int, nargs=5, default=[3]*5)
    parser.add_argument("--batch_norm", default="true")
    parser.add_argument("--dense_layer", type=int, default=128)
    parser.add_argument("--augmentation", default="yes")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--base_dir", default="inaturalist_12K")
    args = parser.parse_args()
    
#Single Training Run:
    
To run :
python DL_Ass2_PartA.py --epochs 20 --batch_size 64 --learning_rate 0.001 --activation gelu

#Hyperparameter Sweep:

python DL_Ass2_PartA.py --sweep --wandb_project naturalist-cnn  --wandb_entity manglesh-patidar-cs23m025


Part A Repo link
Q1:https://github.com/manglesh001/DL_Assignment_2/blob/main/DL_Ass_2_PartA_Q1.ipynb

Q2:https://github.com/manglesh001/DL_Assignment_2/blob/main/DLL_Ass_2_PartA_Q2.ipynb

Q3: No CODE only Obeservation

Q4: https://github.com/manglesh001/DL_Assignment_2/blob/main/dll-ass2-parta-q4.ipynb

Q5: https://github.com/manglesh001/DL_Assignment_2/blob/main/DL_Ass2_PartA.py





## Part B
Use Different Pretrained Model : ResNet50, VGG, GoogleNet,InceptionNet
in this ResNet50 perform better

## # different Straategies used 
1. Freezing All Layers Except the Last Layer
2. Freezing  first k (5) Layers and Fine tunning  the Rest
3. Training all the layer:

  # Wandb arguments
    parser.add_argument("--wandb_entity", "-we", default="manglesh-patidar-cs24m025")
    parser.add_argument("--wandb_project", "-wp", default="inaturalist-classification")
    
    # Training strategy
    parser.add_argument("--strategy", "-s", 
                       choices=['last_layer', 'freeze_k', 'full_model'],
                       default='last_layer',
                       help="Fine-tuning strategy")
    parser.add_argument("--freeze_k", "-k", type=int, default=5,
                       help="Number of initial layers to freeze for freeze_k strategy")
    
    # Model hyperparameters
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--classifier_hidden_units", type=int, default=512)
    
    # Training parameters
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--optimizer", "-o", 
                       choices=['adam', 'nadam', 'rmsprop', 'adamw'],
                       default='adamw')
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "-w_d", type=float, default=1e-4)
    parser.add_argument("--scheduler_factor", type=float, default=0.1)
    parser.add_argument("--scheduler_patience", type=int, default=2)
    
    # Data parameters
    parser.add_argument("--base_dir", "-br", default="inaturalist_12K")


#Example
python DL_Ass_2_PartB.py --strategy freeze_k --freeze_k 3 --optimizer adamw --lr 1e-4

Three Strategies Supported:

last_layer: Freeze all layers except final classifier
python  DL_Ass_2_PartB.py --strategy last_layer --batch_size 64

freeze_k: Freeze first K layers (specify with --freeze_k)
python  DL_Ass_2_PartB.py --strategy freeze_k --freeze_k 5 --learning_rate 1e-4

full_model: Train entire network
python  DL_Ass_2_PartB.py --strategy full_model --learning_rate 1e-5 --batch_size 32


##Part B Repo link

Q1: NO CODE

Q2 & Q3 :  3 Strategies
Freezing All Layers Except the Last Layer: 
https://github.com/manglesh001/DL_Assignment_2/blob/main/dl-ass2-partb-q1.ipynb

Freezing  first k (5) Layers and Fine tunning  the Rest:
https://github.com/manglesh001/DL_Assignment_2/blob/main/dll-ass2-partb-q2b.ipynb

Training All Layers:
https://github.com/manglesh001/DL_Assignment_2/blob/main/dl-ass2-partb-q2c%20(1).ipynb


Q4:  https://github.com/manglesh001/DL_Assignment_2/blob/main/DL_Ass_2_PartB.py



Report Link: https://api.wandb.ai/links/manglesh-patidar-cs24m025/dvbcg3wi


