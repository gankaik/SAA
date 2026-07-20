import argparse

class Options():
    def initialize(self, parser):
        parser.add_argument('--experiment_name', default='my_experiment', help='the name of the experiment')
        parser.add_argument('--seed', type=int, default=3407)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--train_data_root', default='')
        parser.add_argument('--train_classes', nargs='+', default=['car', 'cat', 'chair', 'horse'],
                            help='The image categories included in the training set')
        parser.add_argument('--val_data_root', default='')
        parser.add_argument('--val_classes', nargs='+', default=['car', 'cat', 'chair', 'horse'],
                            help='The image categories included in the validation set')
        parser.add_argument('--max_samples', type=int, default=200, help='Maximum number of samples to use from the dataset (for debugging or limited resources)')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--rank', type=int, default=8, help='LoRA rank for stage 2 training')
        parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter for MixStyle (if applicable)')
        parser.add_argument('--target_layer_idx', nargs='+', type=int, default=[5,6,7,8,9], help='The index of the CLIP layer to apply MixStyle (if applicable)')
        # evaluation setting
        parser.add_argument('--weights', default='', help='trained model weights')
        parser.add_argument('--eval_data_root', default='')


        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0.5)
        parser.add_argument('--blur_sig', default='0.0,3.0')
        parser.add_argument('--jpg_prob', type=float, default=0.5)
        parser.add_argument('--jpg_method', default='cv2,pil')
        parser.add_argument('--jpg_qual', default='30,100')
        parser.add_argument('--use_data_augument',action='store_true')

        return parser

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        self.opt = parser.parse_args()

        return self.opt


    def print_options(self):
        print("----------- Configuration Options -----------")
        for k, v in sorted(vars(self.opt).items()):
            print(f"{k}: {v}")
        print("---------------------------------------------")
