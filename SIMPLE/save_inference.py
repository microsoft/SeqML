# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Import necessary libraries
import os 
import json 
import torch 
from tqdm import tqdm
import miscellaneous.clip as clip
from utils import walkFile, check_file_number

# This function saves the inference results of the trained models
def inference_saver(args, ensemble_net, model_pool):
  
    # Dictionary to store the number of files
    file_num_dic = {}

    # Convert each class name in ensemble_net to a tokenized text input
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in ensemble_net.class_name]).to(args.device)
    print('pretrained_models: ', args.pretrained_models)

    # List of failed models
    failed_list = []

    # Loop through all models to infer results
    for model_idx in range(args.model_num):
      
        # Get the name of the current model
        model_name = args.pretrained_models[model_idx].replace('/','-')
        print('model_idx: ', model_idx, ' model_name:', model_name)

        # Check if the model directory already exists
        if model_name[:5] != '(MAE)':
            model_dir = '/domainbed_inference_saved_' + args.domainbed_dataset + '/' + model_name + '/'
        else:
            model_dir = '/domainbed_inference_saved_' + args.domainbed_dataset + '/' + model_name.split(':')[0] + '/'
        ifexist_model_dir = '/home/private_user_1/v-liziyue/ziyue' + model_dir

        # Create a dictionary to store the number of files for the current model
        file_num_dic[model_name] = {}
        file_num_dic[model_name][args.domainbed_dataset] = walkFile(ifexist_model_dir)
        file_num_dic[model_name]['model_idx'] = model_idx

        # If model directory exists and the files have already been infered, skip the current model
        if os.path.isdir(ifexist_model_dir) and check_file_number(ifexist_model_dir, args.domainbed_dataset):
            print('already available, check passed ', file_num_dic[model_name][args.domainbed_dataset])
            continue

        # If model directory does not exist, create it
        if not os.path.isdir(ifexist_model_dir):
            os.makedirs(ifexist_model_dir)

        # Get the active model and set it to evaluation mode
        active_model = model_pool.models[model_idx].to(args.device)
        active_model.eval()

        # Loop through all the dataloaders to infer and save results
        for dataloader_idx in range(len(ensemble_net.full_loader)):
            data_loader = ensemble_net.full_loader[dataloader_idx]
            dataloader_dir = ifexist_model_dir + args.domainbed_dataset + '_' + str(dataloader_idx) + '/'

            # If the dataloader directory does not exist, create it
            if not os.path.isdir(dataloader_dir):
                os.makedirs(dataloader_dir)

            with torch.no_grad():
                with tqdm(total=len(data_loader), desc='Train') as t:
                    for batch_index, (images, labels, idxs, envs) in enumerate(data_loader):

                        # Create dictionaries to store test results
                        test_dic = {}
                        test_dic_aux = {}
                        images, labels = images.to(args.device), labels.to(args.device)

                        # If the current model is a clip model, encode the image and text features
                        if args.pretrained_models[model_idx][:6] == '(clip)':
                            image_features = active_model.encode_image(images)
                            image_features /= image_features.norm(dim=-1, keepdim=True)

                            text_features = active_model.encode_text(text_inputs)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            
                            out = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                        # If the current model is a swag model, apply the adaptive average pooling after feeding the images to the model
                        elif args.pretrained_models[model_idx][:6] == '(swag)':
                            out = active_model(images)
                            if len(out.shape) == 4:
                                out = torch.nn.AdaptiveAvgPool2d((1,1))(out).squeeze(-1).squeeze(-1)

                        # If the current model is not a clip or swag model, directly feed the images to the model
                        else:
                            out = active_model(images)

                        # Save the results in the dictionaries
                        if type(out) == tuple:

                            for specific_idx, item in enumerate(range(idxs.size(0))):
                                test_dic[item] = out[0][specific_idx,:].cpu()
                                test_dic_aux[item] = out[1][specific_idx,:].cpu()

                            torch.save(test_dic_aux, dataloader_dir + str(batch_index) +  "_aux.pth")
                            output_dim = out[0][specific_idx,:].shape[-1]
                        else:

                            for specific_idx, item in enumerate(range(idxs.size(0))):
                                test_dic[item] = out[specific_idx,:].cpu()
                            output_dim = out[specific_idx,:].shape[-1]

                        torch.save(test_dic, dataloader_dir + str(batch_index) +  ".pth")

                        t.set_postfix({
                                'test_batch': batch_index,
                            })
                        t.update(1)

        # Record the number of files that were infered for the current model
        file_num_dic[model_name] = {}
        file_num_dic[model_name][args.domainbed_dataset] = walkFile(ifexist_model_dir)
        file_num_dic[model_name]['model_idx'] = model_idx
        file_num_dic[model_name]['output_dim'] = output_dim

    # Print the failed models as well as the number of files that were infered
    print('failed_list:')
    print(failed_list)

    print('file num count:')
    print(file_num_dic)

    # Save the number of files dictionary to a json file
    save_dir = os.environ.get('AMLT_OUTPUT_DIR', '.')
    with open(f"{save_dir}/{args.domainbed_dataset}_file_num_dic.json", "w") as f:
        json.dump(file_num_dic, f, indent=4, sort_keys=True)