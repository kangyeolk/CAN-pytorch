import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():
    parser = argparse.ArgumentParser()

    # Training dataset
    parser.add_argument('--dataset', required=True, help='two class of dog\'s kind or other dataset')
    parser.add_argument('--exec_data_setter', type=str2bool, default=True, help='Shell script for temporary fixing')

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='can', choices=['can'])
    parser.add_argument('--img_size', type=int, default=256, help='Input image size') # data image size
    parser.add_argument('--z_size', type=int, default=100, help='Latent vector z size')
    parser.add_argument('--n_class', type=int, default=2, help='Number of class')

    # Traning setting
    parser.add_argument('--n_steps', type=int, default=100000, help='Number of steps to update generator and discriminator')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--beta0', type=float, default=0.9, help='Adam optimizer parameter-1')
    parser.add_argument('--beta1', type=float, default=0.999, help='Adam optimizer parameter-2')
    parser.add_argument('--slope', type=int, default=0.2, help='Slope of Leaky ReLU')
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--img_rootpath', type=str, default='./data/Images')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--model_save_path', type=str, default='./model')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)

    return parser.parse_args()
