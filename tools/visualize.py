import argparse
import os
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

# import sys
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/../")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize inference results on images')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--imgs_dir', 
        help='images directory',
        default='./examples')
    parser.add_argument(
        '--show_dir', 
        help='directory where painted images will be saved')
    
    args = parser.parse_args()
    return args

    
def main():
    args = parse_args()
    
    device = 'cuda:0'
    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = None
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']
    
    # set the model's cfg for inference
    model.cfg = config
    
    model.to(device)
    model.eval()
    
    imgs_dir = args.imgs_dir + '/'
    show_dir = args.show_dir + '/'
    
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    for img in os.listdir(imgs_dir):
        result = inference_detector(model,imgs_dir+img)
        show_result_pyplot(model, imgs_dir+img, result, out_file=show_dir+img)

if __name__ == '__main__':
    main()