import argparse, os
import torch
from utils.helpers import makedir
import utils.save as save
from utils.log import create_logger

from cfg.default import get_cfg_defaults
from preprocessing.datasets import prepare_dataset, get_dataloaders
from preprocessing.cub import preprocess_cub_input_function

from model.model import construct_ppnet
import traintest.train_and_test as tnt

import postprocessing.push as push
import postprocessing.prune as prune

def update_cfg(cfg, args): 
    if cfg.MODEL.DEVICE == "cuda": 
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
        print(f"Using GPU : {os.environ['CUDA_VISIBLE_DEVICES']}")

    # first update model 
    cfg.MODEL.BACKBONE = args.backbone
    cfg.MODEL.PROTOTYPE_ACTIVATION_FUNCTION = "log" 
    cfg.MODEL.USE_COSINE = False 

    if args.dataset == "cub": 

        cfg.MODEL.PROTOTYPE_SHAPE = (2000, 128, 1, 1) 
        cfg.MODEL.ADD_ON_LAYERS_TYPE = "regular"
        
        cfg.DATASET.NAME = "cub"
        cfg.DATASET.NUM_CLASSES = 200 
        cfg.DATASET.IMAGE_SIZE = 224
        cfg.DATASET.DATA_PATH = os.path.join("data", "CUB_200_2011", "cub200_cropped")
        cfg.DATASET.TRAIN_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped_augmented")
        cfg.DATASET.TEST_DIR = os.path.join(cfg.DATASET.DATA_PATH, "test_cropped")
        cfg.DATASET.TRAIN_PUSH_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped")
        cfg.DATASET.TRAIN_BATCH_SIZE = 80
        cfg.DATASET.TRANSFORM_MEAN = (0.485, 0.456, 0.406) 
        cfg.DATASET.TRANSFORM_STD = (0.229, 0.224, 0.225)

    elif args.dataset == "bioscan":
        cfg.MODEL.PROTOTYPE_SHAPE = (40 * 40, 128, 1, 1) 
        cfg.MODEL.ADD_ON_LAYERS_TYPE = None 
        
        cfg.DATASET.NAME = "bioscan"
        cfg.DATASET.BIOSCAN.TAXONOMY_NAME = "family"
        cfg.DATASET.BIOSCAN.ORDER_NAME = "Diptera"
        cfg.DATASET.BIOSCAN.CHOP_LENGTH = 720 
        cfg.DATASET.NUM_CLASSES = 40
        cfg.DATASET.IMAGE_SIZE = (4, 1, cfg.DATASET.BIOSCAN.CHOP_LENGTH)
        # cfg.DATASET.DATA_PATH = os.path.join("data", "CUB_200_2011", "cub200_cropped")
        # cfg.DATASET.TRAIN_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped_augmented")
        # cfg.DATASET.TEST_DIR = os.path.join(cfg.DATASET.DATA_PATH, "test_cropped")
        # cfg.DATASET.TRAIN_PUSH_DIR = os.path.join(cfg.DATASET.DATA_PATH, "train_cropped")
        cfg.DATASET.TRAIN_BATCH_SIZE = 80

    else: 
        raise Exception("Invalid Dataset")

    model_dir = os.path.join("saved_models", f"{cfg.DATASET.NAME}_ppnet", str(cfg.EXPERIMENT_RUN).zfill(3))
    while os.path.isdir(model_dir): 
        cfg.EXPERIMENT_RUN += 1
        model_dir = os.path.join("saved_models", f"{cfg.DATASET.NAME}_ppnet", str(cfg.EXPERIMENT_RUN).zfill(3))
    makedir(model_dir)

    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    print(f"Model Dir: {model_dir}")

    cfg.OUTPUT.MODEL_DIR = model_dir
    cfg.OUTPUT.IMG_DIR = img_dir 
    cfg.OUTPUT.WEIGHT_MATRIX_FILENAME = weight_matrix_filename 
    cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX = prototype_img_filename_prefix 
    cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX = prototype_self_act_filename_prefix 
    cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX = proto_bound_boxes_filename_prefix 
     
def get_optimizers(cfg, ppnet): 
    joint_optimizer_specs = [
        {
            'params': ppnet.features.parameters(), 
            'lr': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.FEATURES, 
            'weight_decay': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.WEIGHT_DECAY
        }, # bias are now also being regularized
        {
            'params': ppnet.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.ADD_ON_LAYERS,
            'weight_decay': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.WEIGHT_DECAY
        },
        {
            'params': ppnet.prototype_vectors, 
            'lr': cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.PROTOTYPE_VECTORS
        },
        ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=cfg.OPTIM.JOINT_OPTIMIZER_LAYERS.LR_STEP_SIZE, gamma=0.1)

    warm_optimizer_specs = [
        {
            'params': ppnet.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.WARM_OPTIMIZER_LAYERS.ADD_ON_LAYERS,
            'weight_decay': cfg.OPTIM.WARM_OPTIMIZER_LAYERS.WEIGHT_DECAY
        },
        {'params': ppnet.prototype_vectors, 
         'lr': cfg.OPTIM.WARM_OPTIMIZER_LAYERS.PROTOTYPE_VECTORS,
         },
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    last_layer_optimizer_specs = [
        {
            'params': ppnet.last_layer.parameters(), 
            'lr': cfg.OPTIM.LAST_LAYER_OPTIMIZER_LAYERS.LR
        }
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    return joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer

if __name__ == '__main__':

    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--backbone', type=str, default='')
    args = parser.parse_args()

    # prepare dataset such as augmentation if not done yet
    prepare_dataset(args)
    # update the hyperparameters from default to the ones we mentioned in arguments
    update_cfg(cfg, args) 

    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))

    log(str(cfg))
    print(cfg)
    train_loader, train_push_loader, test_loader = get_dataloaders(cfg, log)

    ppnet = construct_ppnet(cfg)

    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True
    
    joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(cfg, ppnet)

    log('start training')
    
    coefs = {
        'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
        'clst': cfg.OPTIM.COEFS.CLST,
        'sep': cfg.OPTIM.COEFS.SEP,
        'l1': cfg.OPTIM.COEFS.L1,
    }

    for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
        log('epoch: \t{0}'.format(epoch))

        if epoch < cfg.OPTIM.NUM_WARM_EPOCHS:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.70, log=log)

        if epoch >= cfg.OPTIM.PUSH_START and epoch in cfg.OPTIM.PUSH_EPOCHS:
            
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_cub_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
                prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
                proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
                save_prototype_class_identity=True,
                log=log)
            
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=0.70, log=log)

            if cfg.MODEL.PROTOTYPE_ACTIVATION_FUNCTION != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu, target_accu=0.70, log=log)
       
    logclose()

