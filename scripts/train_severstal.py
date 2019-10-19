from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

# Set up paths
train_pd = pd.read_csv('/root/.fastai/data/severstal/train.csv')
path = Path('/root/.fastai/data/severstal')
train_images = get_image_files(path/'train_images')

mask_path = Path('/kaggle/mask')
if not os.path.exists(mask_path):
    os.makedirs(str(mask_path))
    
# Functions for encoding the image

def convert_encoded_to_array(encoded_pixels):
    pos_array = []
    len_array = []
    splits = encoded_pixels.split()
    pos_array = [int(n) - 1 for i, n in enumerate(splits) if i % 2 == 0]
    len_array = [int(n) for i, n in enumerate(splits) if i % 2 == 1]
    return pos_array, len_array
        
def convert_to_pair(pos_array, rows):
    return [(p % rows, p // rows) for p in pos_array]

def create_positions(single_pos, size):
    return [i for i in range(single_pos, single_pos + size)]

def create_positions_pairs(single_pos, size, row_size):
    return convert_to_pair(create_positions(single_pos, size), row_size)

def convert_to_mask(encoded_pixels, row_size, col_size, category):
    pos_array, len_array = convert_encoded_to_array(encoded_pixels)
    mask = np.zeros([row_size, col_size])
    for(p, l) in zip(pos_array, len_array):
        for row, col in create_positions_pairs(p, l, row_size):
            mask[row][col] = category
    return mask

def save_to_image(masked, image_name):
    im = PIL.Image.fromarray(masked)
    im = im.convert("L")
    image_name = re.sub(r'(.+)\.jpg', r'\1', image_name) + ".png"
    real_path = mask_path/image_name
    im.save(real_path)
    return real_path
    
def get_y_fn(x):
    return mask_path/(x.stem + '.png')

def group_by(train_images, train_pd):
    tran_dict = {image.name:[] for image in train_images}
    pattern = re.compile('(.+)_(\d+)')
    for index, image_path in train_pd.iterrows():
        m = pattern.match(image_path['ImageId_ClassId'])
        file_name = m.group(1)
        category = m.group(2)
        tran_dict[file_name].append((int(category), image_path['EncodedPixels']))
    return tran_dict

grouped_categories_mask = group_by(train_images, train_pd)

### Create mask files and save these to kaggle/mask/

image_height = 256
image_width = 1600
if not os.path.exists(mask_path/'0002cc93b.png'):
    for image_name, cat_list in grouped_categories_mask.items():
        masked = np.zeros([image_height, image_width])
        for cat_mask in cat_list:
            encoded_pixels = cat_mask[1]
            if pd.notna(cat_mask[1]):
                masked += convert_to_mask(encoded_pixels, image_height, image_width, cat_mask[0])
        if np.amax(masked) > 4:
            print(f'Check {image_name} for max category {np.amax(masked)}')
        save_to_image(masked, image_name)
        
### Transforms

def get_simple_transforms(max_rotate:float=3., max_zoom:float=1.1,
                   max_lighting:float=0.2, max_warp:float=0.2, p_affine:float=0.75,
                   p_lighting:float=0.75, xtra_tfms:Optional[Collection[Transform]]=None)->Collection[Transform]:
    "Utility func to easily create a list of flip, rotate, `zoom`, warp, lighting transforms."
    res = [
        rand_crop(),
        symmetric_warp(magnitude=(-max_warp,max_warp), p=p_affine),
        rotate(degrees=(-max_rotate,max_rotate), p=p_affine),
        rand_zoom(scale=(1., max_zoom), p=p_affine)
          ]
    #       train                   , valid
    return (res, [crop_pad()])

# Setting up variables
train_images = (path/'train_images').ls()
src_size = np.array(open_image(str(train_images[0])).shape[1:])
valid_pct = 0.10

codes = array(['0', '1', '2', '3', '4'])

# Create data bunch

def create_data_bunch(bs, size):
    src = (SegmentationItemList.from_folder(path/'train_images')
       .split_by_rand_pct(valid_pct=valid_pct)
       .label_from_func(get_y_fn, classes=codes))
    data = (src.transform(get_simple_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
    return src, data

bs = 4
size = src_size//2
src, data = create_data_bunch(bs, size)

# Metric functions

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['0']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    argmax = (input.argmax(dim=1))
    comparison = argmax[mask]==target[mask]
    return torch.tensor(0.) if comparison.numel() == 0 else comparison.float().mean()

def acc_camvid_with_zero_check(input, target):
    target = target.squeeze(1)
    argmax = (input.argmax(dim=1))
    batch_size = input.shape[0]
    total = torch.empty([batch_size])
    for b in range(batch_size):
        if(torch.sum(argmax[b]).item() == 0.0 and torch.sum(target[b]).item() == 0.0):
            total[b] = 1
        else:
            mask = target[b] != void_code
            comparison = argmax[b][mask]==target[b][mask]
            total[b] = torch.tensor(0.) if comparison.numel() == 0 else comparison.float().mean()
    return total.mean()


def calc_dice_coefficients(argmax, target, cats):
    def calc_dice_coefficient(seg, gt, cat: int):
        mask_seg = seg == cat
        mask_gt = gt == cat
        sum_seg = torch.sum(mask_seg.float())
        sum_gt = torch.sum(mask_gt.float())
        if sum_seg + sum_gt == 0:
            return torch.tensor(1.0)
        return (torch.sum((seg[gt == cat] / cat).float()) * 2.0) / (sum_seg + sum_gt)

    total_avg = torch.empty([len(cats)])
    for i, c in enumerate(cats):
        total_avg[i] = calc_dice_coefficient(argmax, target, c)
    return total_avg.mean()


def dice_coefficient(input, target):
    target = target.squeeze(1)
    argmax = (input.argmax(dim=1))
    batch_size = input.shape[0]
    cats = [1, 2, 3, 4]
    total = torch.empty([batch_size])
    for b in range(batch_size):
        total[b] = calc_dice_coefficients(argmax[b], target[b], cats)
    return total.mean()

def calc_dice_coefficients_2(argmax, target, cats):
    def calc_dice_coefficient(seg, gt, cat: int):
        mask_seg = seg == cat
        mask_gt = gt == cat
        sum_seg = torch.sum(mask_seg.float())
        sum_gt = torch.sum(mask_gt.float())
        return (torch.sum((seg[gt == cat] / cat).float())), (sum_seg + sum_gt)

    total_avg = torch.empty([len(cats), 2])
    for i, c in enumerate(cats):
        total_avg[i][0], total_avg[i][1] = calc_dice_coefficient(argmax, target, c)
    total_sum = total_avg.sum(axis=0)
    if (total_sum[1] == 0.0):
        return torch.tensor(1.0)
    return total_sum[0] * 2.0 / total_sum[1]


def dice_coefficient_2(input, target):
    target = target.squeeze(1)
    argmax = (input.argmax(dim=1))
    batch_size = input.shape[0]
    cats = [1, 2, 3, 4]
    total = torch.empty([batch_size])
    for b in range(batch_size):
        total[b] = calc_dice_coefficients_2(argmax[b], target[b], cats)
    return total.mean()


def accuracy_simple(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

### Customized loss function

class CombinedDiceLoss(nn.Module):
    def __init__(self, zero_cat_factor=0.1):
        super().__init__()
        self.zero_cat_factor = zero_cat_factor

    def forward(self, input, target):
        return self.dice_loss(target, input, self.zero_cat_factor)

    def dice_loss(self, target, output, eps=1e-7, zero_cat_factor=0.1):
        '''
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.

        # Arguments
            target: b x 1 x X x Y( x Z...) ground truth
            output: b x c x X x Y( x Z...) Network output, must sum to 1 over c channel (such as after softmax)
            epsilon: Used for numerical stability to avoid divide by zero errors

        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        '''

        # skip the batch and class axis for calculating Dice score
        num_classes = output.shape[1]
        y_true = F.one_hot(target.long().squeeze(), num_classes)
        y_pred = F.softmax(output, dim=1).permute(0, 2, 3, 1)
        y_true = y_true.type(y_pred.type())
        y_true = y_true.permute(0, 3, 1, 2)
        y_true[:,0,:] *= zero_cat_factor # Factor used to take power away from the zeroth category
        y_true = y_true.permute(0, 2, 3, 1)
        axes = tuple(range(1, len(y_pred.shape)-1))
        numerator = 2. * torch.sum(y_pred * y_true, axes)
        denominator = torch.sum(y_pred ** 2 + y_true ** 2, axes)
        # When intersection and cardinality are all zero you have 100% score and not 0% score
        # For this we use the eps parameter
        loss_array = ((numerator + eps) / (denominator + eps))
        
        
        loss_array = (loss_array).mean(dim=0)
        return ((1 - torch.mean(loss_array)) + F.cross_entropy(output, target.squeeze())) / 2.

    def __del__(self): pass
    
##### The main training function

from fastai import callbacks

def train_learner(learn, slice_lr, epochs=10, pct_start=0.8, best_model_name='best_model', 
                  patience_early_stop=4, patience_reduce_lr = 3):
    learn.fit_one_cycle(epochs, slice_lr, pct_start=pct_start, 
                    callbacks=[callbacks.SaveModelCallback(learn, monitor='dice_coefficient',mode='max', name=best_model_name),
                              callbacks.EarlyStoppingCallback(learn=learn, monitor='dice_coefficient', patience=patience_early_stop),
                              callbacks.ReduceLROnPlateauCallback(learn=learn, monitor='dice_coefficient', patience=patience_reduce_lr),
                              callbacks.TerminateOnNaNCallback()])
    
### First Training

metrics=accuracy_simple, acc_camvid_with_zero_check, dice_coefficient, dice_coefficient_2
wd=1e-2

learn = unet_learner(data, models.resnet50, metrics=metrics, wd=wd, bottle=True)
learn.loss_func = CombinedDiceLoss(zero_cat_factor=0.5)
print(learn.loss_func)

learn.model_dir = Path('/kaggle/model')
learn = to_fp16(learn, loss_scale=4.0)

lr_find(learn, num_it=400)
learn.recorder.plot()
    
