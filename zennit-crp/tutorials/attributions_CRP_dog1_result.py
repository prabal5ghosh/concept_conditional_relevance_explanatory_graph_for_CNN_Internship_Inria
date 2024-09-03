# %%
import torch

import sys
import os

from crp.concepts import ChannelConcept

cc = ChannelConcept()


import torch
from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
from PIL import Image


import torch
from crp.attribution import CondAttribution
from crp.image import imgify
import numpy as np
from PIL import Image, ImageDraw, ImageFont





# %%
import torch

import sys
import os

from crp.concepts import ChannelConcept

cc = ChannelConcept()


import torch
from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
from PIL import Image


import torch
from crp.attribution import CondAttribution
from crp.image import imgify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.models.vgg import vgg16_bn
import torchvision.transforms as T
from PIL import Image
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = vgg16_bn(True).to(device)
model.eval()

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
image = Image.open("images/dog1.jpg")

sample = transform(image).unsqueeze(0).to(device)

import sys

output_file_path = '/home/prghosh/output_image/image_512/output_file.txt'


with open(output_file_path, 'w') as file:
    # Redirect stdout to the file
    sys.stdout = file
    
        
    print("shape of the image: ",sample.shape)
    image


    print("--------------modified image----------------------")


    # # Define the region of interest (ROI)
    # x_start, y_start = 200, 90  # Top-left corner of the ROI
    # width, height = 150, 60     # Width and height of the ROI

    # # Convert the image to a NumPy array
    # image_array = np.array(image)

    # # Create a white image of the same size
    # white_image_array = np.ones_like(image_array) * 255  # RGB value for white is [255, 255, 255]

    # # Overlay the original ROI onto the white image
    # white_image_array[y_start:y_start+height, x_start:x_start+width] = image_array[y_start:y_start+height, x_start:x_start+width]

    # # Convert back to a PIL image
    # modified_image = Image.fromarray(white_image_array)


    # # Save the modified image
    # modified_image.save( "/home/prghosh/output_image/image_512/modified_image.jpg")  # Change the file name and format as needed

    # # Apply PyTorch transformations
    # transform = T.Compose([
    #     T.Resize(256),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # # Transform the modified image
    # sample_t = transform(modified_image).unsqueeze(0).to(device)

    # # Now `sample` contains the transformed tensor ready for use with models
    # print(sample_t.shape)




    # Unnormalize the tensor for visualization
    unnormalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    sample_unnorm = unnormalize(sample.squeeze(0))

    # Convert the tensor to a numpy array
    sample_np = sample_unnorm.permute(1, 2, 0).detach().cpu().numpy()
    sample_np = np.clip(sample_np, 0, 1)  # Clip values to be between 0 and 1

    # Convert to uint8 format for PIL compatibility
    sample_np_uint8 = (sample_np * 255).astype(np.uint8)

    # Plot the image
    # plt.imshow(sample_np_uint8)
    # plt.title('Plot of Transformed Tensor `sample` with size 224*224')
    # plt.show()

    from PIL import Image
    import numpy as np

    # Convert the NumPy array to a PIL Image
    image_to_save = Image.fromarray(sample_np_uint8)

    # Save the image to a file
    image_to_save.save("/home/prghosh/output_image/image_512/sample_224*224.jpg")  # Change the file path and name as needed

    print("modified Image saved successfully!")

    print("--------------mask image----------------------")

    # Define the region of interest (ROI)
    x_start, y_start = 85, 20  # Top-left corner of the ROI
    width, height = 75, 25    # Width and height of the ROI


    # Create a white image of the same size
    white_image_array = np.ones_like(sample_np_uint8) * 255  # RGB value for white is [255, 255, 255]

    # Overlay the original ROI onto the white image
    white_image_array[y_start:y_start+height, x_start:x_start+width] = sample_np_uint8[y_start:y_start+height, x_start:x_start+width]

    # Convert back to a PIL image
    modified_image = Image.fromarray(white_image_array)

    # Plot the image with the ROI
    # plt.imshow(modified_image)

    # Save the image to a file
    modified_image.save( "/home/prghosh/output_image/image_512/masked_image.jpg")  # Change the file name and format as needed

    # image_to_save.save("modified_image.jpg")  # Change the file path and name as needed

    print("masked_image.jpg saved successfully!")




    from zennit.composites import EpsilonPlusFlat   #  computes LRP relevance
    from zennit.canonizers import SequentialMergeBatchNorm   # Canonizer to merge the parameters of all batch norms that appear sequentially right after a linear module.

    from crp.attribution import CondAttribution  # This class contains the functionality to compute conditional attributions.

    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])  # The EpsilonPlusFlat composite will be used with a SequentialMergeBatchNorm to handle batch normalization layers.
    attribution = CondAttribution(model, no_param_grad=True)   # CondAttribution(model, no_param_grad=True) creates an attribution object for model without calculating gradients with respect to its parameters.



    sample.requires_grad = True

    from crp.image import imgify
    from crp.helper import get_layer_names
    from crp.attribution import CondAttribution
    from crp.image import imgify

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])

    print("layer_names: ",layer_names)
    file_name =["heatmap_grid_00.jpg","heatmap_grid_01.jpg","heatmap_grid_02.jpg","heatmap_grid_03.jpg","heatmap_grid_04.jpg","heatmap_grid_05.jpg","heatmap_grid_06.jpg","heatmap_grid_07.jpg","heatmap_grid_08.jpg","heatmap_grid_09.jpg","heatmap_grid_10.jpg","heatmap_grid_11.jpg","heatmap_grid_12.jpg","heatmap_grid_13.jpg","heatmap_grid_14.jpg","heatmap_grid_15.jpg", "heatmap_grid_16.jpg"]

    file_mask_name = ["mask0.jpg","mask1.jpg","mask2.jpg"]



    file_name_new = [
        "pixel_features_40.jpg", "pixel_features_37.jpg", "pixel_features_34.jpg", "pixel_features_30.jpg", "pixel_features_27.jpg", 
        "pixel_features_24.jpg", "pixel_features_20.jpg", "pixel_features_17.jpg", "pixel_features_14.jpg", "pixel_features_10.jpg", 
        "pixel_features_7.jpg", "pixel_features_3.jpg", "pixel_features_0.jpg", "pixel_13.jpg", "pixel_14.jpg", 
        "pixel_15.jpg","pixel_16.jpg"
    ]

    relevance_plot = ['relevance_val_top_10_percent_40.jpg', 'relevance_val_top_10_percent_37.jpg', 'relevance_val_top_10_percent_34.jpg', 'relevance_val_top_10_percent_30.jpg', 'relevance_val_top_10_percent_27.jpg', 'relevance_val_top_10_percent_24.jpg', 'relevance_val_top_10_percent_20.jpg', 'relevance_val_top_10_percent_17.jpg', 'relevance_val_top_10_percent_14.jpg', 'relevance_val_top_10_percent_10.jpg', 'relevance_val_top_10_percent_7.jpg', 'relevance_val_top_10_percent_4.jpg', 'relevance_val_top_10_percent_0.jpg']

    # %%
    from crp.helper import get_layer_names
    from crp.attribution import CondAttribution
    from crp.image import imgify

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])

    import matplotlib.pyplot as plt







    # conditions = [{'y': [208]}]

    # attr = attribution(sample, conditions, composite, record_layer=layer_names)

    # attr.activations['features.40'].shape, attr.relevances['features.40'].shape
    # # attr[1]["features.40"].shape, attr[2]["features.40"].shape # is equivalent

    # # %%
    # # layer features.40 has 512 channel concepts
    # rel_c = cc.attribute(attr.relevances['features.40'], abs_norm=True)
    # rel_c.shape

    # # %%
    # # the six most relevant concepts and their contribution to final classification in percent
    # rel_values, concept_ids = torch.topk(rel_c[0], 25)
    # print("concept_ids:",concept_ids)
    # print(rel_values*100)

    # # %%
    # if type(concept_ids) != list : 
    #     concept_ids = concept_ids.tolist()
    # concept_ids

    # # %%
    # print('concept_id:',concept_ids)







    print("---------------------------------1----------------------------------------------------")
    print("-------------------------------------------------------------------------------------")




    # %%
    conditions = [{'features.40': [id], 'y': [208]} for id in torch.arange(0, 512)]

    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     pass





    # %%
    mask = torch.zeros(224, 224).to(attribution.device)
    mask[y_start:y_start+height, x_start:x_start+width] = 1

    final_image = imgify(mask, symmetric=True)



    # Convert to 'RGB' mode if necessary
    if final_image.mode != 'RGB':
        final_image = final_image.convert('RGB')

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_mask_name[0]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Print the filename
    print(f"The mask is saved as  {file_mask_name[0]}")








    print("--------------------------------layer40--------------------------------------------------------")
    print("--------------------------------layer40--------------------------------------------------------")


    print("-------------------------------2------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")











    # %%

    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)



    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list)





    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 40 --len_10_percent_indices",len_10_percent_indices)
    print("layer 40 --10_percent_rel_indices",indices_of_true)
    print("layer 40 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[0]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 40 --indices of top 10% rel_c----: ",indices)
    print("layer 40 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)



    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list = concept_ids.tolist()

    # Print the resulting list
    print("layer 40 --concept_ids_list : ",concept_ids_list)

    #################################################################################################

    print("-------------------------------3------------------------------------------------------")
    print("-------------------------------------------------------------------------------------")










    # # %%



    # # # heatmap_s=[]
    # # # for i in concept_ids_list:
    # # #     conditions = [{'features.40': [i], 'y': [208]}]
    # # #     heatmap, _, _, _ = attribution(sample, conditions, composite)
    # # #     heatmap_s.append(heatmap)
        
    # # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # # print(conditions)
    # # heatmap, _, _, _ = attribution(sample, conditions, composite)

    # # final_image = imgify(heatmap, symmetric=True, grid=(1, 5))


    # # # Convert to 'RGB' mode if necessary
    # # if final_image.mode != 'RGB':
    # #     final_image = final_image.convert('RGB')

    # # # Specify the path and filename
    # # save_directory = "/home/prghosh/output_image/image_512/"
    # # if not os.path.exists(save_directory):
    # #     os.makedirs(save_directory)

    # # file_name = file_name_new[0]
    # # file_path = os.path.join(save_directory, file_name)

    # # # Save the image as a JPG file
    # # final_image.save(file_path, format="JPEG")

    # # # Print the filename
    # # print(f"Saved heatmap grid as {file_name_new[0]}")













    heatmap_s=[]
    for i in concept_ids_list:
        conditions = [{'features.40': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5, 8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[0]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[0]}")











    print("..................layer -37-----------------------------")
    print("..................layer -37-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': [id], 'y': [208]} for id in torch.arange(0, 512)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_1 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_1)







    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 37 --len_10_percent_indices",len_10_percent_indices)
    print("layer 37 --10_percent_rel_indices",indices_of_true)
    print("layer 37 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[1]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 37 --indices of top 10% rel_c----: ",indices)
    print("layer 37 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_1 = concept_ids.tolist()

    # Print the resulting list
    print("layer 37 --concept_ids_list : ",concept_ids_list_1)


    #################################################################################################








    heatmap_s=[]
    for i in concept_ids_list_1:
        conditions = [{'features.40': concept_ids_list,'features.37': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[1]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[1]}")












    print("..................layer -34-----------------------------")
    print("..................layer -34-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': [id], 'y': [208]} for id in torch.arange(0, 512)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_2 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_2)






    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 34 --len_10_percent_indices",len_10_percent_indices)
    print("layer 34 --10_percent_rel_indices",indices_of_true)
    print("layer 34 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[2]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 34 --indices of top 10% rel_c----: ",indices)
    print("layer 34 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_2 = concept_ids.tolist()

    # Print the resulting list
    print("layer 34 --concept_ids_list : ",concept_ids_list_2)


    #################################################################################################






    heatmap_s=[]
    for i in concept_ids_list_2:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[2]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[2]}")
















    print("..................layer -30-----------------------------")
    print("..................layer -30-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': [id], 'y': [208]} for id in torch.arange(0, 512)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_3 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_3)



    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 30 --len_10_percent_indices",len_10_percent_indices)
    print("layer 30 --10_percent_rel_indices",indices_of_true)
    print("layer 30 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[3]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 30 --indices of top 10% rel_c----: ",indices)
    print("layer 30 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_3 = concept_ids.tolist()

    # Print the resulting list
    print("layer 30 --concept_ids_list : ",concept_ids_list_3)


    #################################################################################################











    heatmap_s=[]
    for i in concept_ids_list_3:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[3]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[3]}")











    print("..................layer -27-----------------------------")
    print("..................layer -27-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': [id], 'y': [208]} for id in torch.arange(0, 512)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_4 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_4)





    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 27 --len_10_percent_indices",len_10_percent_indices)
    print("layer 27 --10_percent_rel_indices",indices_of_true)
    print("layer 27 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[4]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 27 --indices of top 10% rel_c----: ",indices)
    print("layer 27 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_4 = concept_ids.tolist()

    # Print the resulting list
    print("layer 27 --concept_ids_list : ",concept_ids_list_4)


    #################################################################################################







    heatmap_s=[]
    for i in concept_ids_list_4:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[4]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[4]}")








    print("..................layer -24-----------------------------")
    print("..................layer -24-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': [id], 'y': [208]} for id in torch.arange(0, 512)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_5 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_5)




    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 24 --len_10_percent_indices",len_10_percent_indices)
    print("layer 24 --10_percent_rel_indices",indices_of_true)
    print("layer 24 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[5]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 24 --indices of top 10% rel_c----: ",indices)
    print("layer 24 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_5 = concept_ids.tolist()

    # Print the resulting list
    print("layer 24 --concept_ids_list : ",concept_ids_list_5)


    #################################################################################################








    heatmap_s=[]
    for i in concept_ids_list_5:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[5]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[5]}")











    print("..................layer -20-----------------------------")
    print("..................layer -20-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': [id], 'y': [208]} for id in torch.arange(0, 256)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_6 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_6)





    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 20 --len_10_percent_indices",len_10_percent_indices)
    print("layer 20 --10_percent_rel_indices",indices_of_true)
    print("layer 20 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[6]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 20 --indices of top 10% rel_c----: ",indices)
    print("layer 20 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_6 = concept_ids.tolist()

    # Print the resulting list
    print("layer 20 --concept_ids_list : ",concept_ids_list_6)


    #################################################################################################







    heatmap_s=[]
    for i in concept_ids_list_6:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[6]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[6]}")












    print("..................layer -17-----------------------------")
    print("..................layer -17-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': [id], 'y': [208]} for id in torch.arange(0, 256)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_7 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_7)




    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 17 --len_10_percent_indices",len_10_percent_indices)
    print("layer 17 --10_percent_rel_indices",indices_of_true)
    print("layer 17 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[7]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 17 --indices of top 10% rel_c----: ",indices)
    print("layer 17 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_7 = concept_ids.tolist()

    # Print the resulting list
    print("layer 17 --concept_ids_list : ",concept_ids_list_7)


    #################################################################################################








    heatmap_s=[]
    for i in concept_ids_list_7:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[7]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[7]}")










    print("..................layer -14-----------------------------")
    print("..................layer -14-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': [id], 'y': [208]} for id in torch.arange(0, 256)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_8 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_8)






    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 14 --len_10_percent_indices",len_10_percent_indices)
    print("layer 14 --10_percent_rel_indices",indices_of_true)
    print("layer 14 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[8]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 14 --indices of top 10% rel_c----: ",indices)
    print("layer 14 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_8 = concept_ids.tolist()

    # Print the resulting list
    print("layer 14 --concept_ids_list : ",concept_ids_list_8)


    #################################################################################################








    heatmap_s=[]
    for i in concept_ids_list_8:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[8]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[8]}")










    print("..................layer -10-----------------------------")
    print("..................layer -10-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': [id], 'y': [208]} for id in torch.arange(0, 128)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_9 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_9)




    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 10 --len_10_percent_indices",len_10_percent_indices)
    print("layer 10 --10_percent_rel_indices",indices_of_true)
    print("layer 10 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[9]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 10 --indices of top 10% rel_c----: ",indices)
    print("layer 10 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_9 = concept_ids.tolist()

    # Print the resulting list
    print("layer 10 --concept_ids_list : ",concept_ids_list_9)


    #################################################################################################








    heatmap_s=[]
    for i in concept_ids_list_9:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[9]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[9]}")










    print("..................layer -7-----------------------------")
    print("..................layer -7-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': concept_ids_list_9,'features.7': [id], 'y': [208]} for id in torch.arange(0, 128)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_10 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_10)






    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 7 --len_10_percent_indices",len_10_percent_indices)
    print("layer 7 --10_percent_rel_indices",indices_of_true)
    print("layer 7 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[10]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 7 --indices of top 10% rel_c----: ",indices)
    print("layer 7 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_10 = concept_ids.tolist()

    # Print the resulting list
    print("layer 7 --concept_ids_list : ",concept_ids_list_10)


    #################################################################################################






    heatmap_s=[]
    for i in concept_ids_list_10:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': concept_ids_list_9,'features.7': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[10]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[10]}")









    print("..................layer -3-----------------------------")
    print("..................layer -3-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': concept_ids_list_9,'features.7': concept_ids_list_10,'features.3': [id], 'y': [208]} for id in torch.arange(0, 64)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_11 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_11)




    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 3 --len_10_percent_indices",len_10_percent_indices)
    print("layer 3 --10_percent_rel_indices",indices_of_true)
    print("layer 3 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[11]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 3 --indices of top 10% rel_c----: ",indices)
    print("layer 3 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_11 = concept_ids.tolist()

    # Print the resulting list
    print("layer 3 --concept_ids_list : ",concept_ids_list_11)


    #################################################################################################








    heatmap_s=[]
    for i in concept_ids_list_11:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': concept_ids_list_9,'features.7': concept_ids_list_10,'features.3': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[11]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[11]}")








    print("..................layer -0-----------------------------")
    print("..................layer -0-----------------------------")



    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': concept_ids_list_9,'features.7': concept_ids_list_10,'features.3': concept_ids_list_11,'features.0': [id], 'y': [208]} for id in torch.arange(0, 64)]


    from crp.helper import abs_norm

    rel_c = []
    for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
        
        masked = attr.heatmap * mask[None, :, :]
        rel_c.append(torch.sum(masked, dim=(1, 2)))

    rel_c = torch.cat(rel_c)

    # indices = torch.topk(rel_c, 25).indices
    # # we norm here, so that we clearly see the contribution inside the masked region as percentage
    # print("indices of top 25 rel_c: ",indices)
    # print("top 25 rel_c with percentage: ",abs_norm(rel_c)[indices]*100)


    # import torch

    # # Your tensor
    # concept_ids = indices

    # # Convert the tensor to a list
    # concept_ids_list_12 = concept_ids.tolist()

    # # Print the resulting list
    # print(concept_ids_list_12)





    #########################################################################################################################""

    import torch
    from crp.helper import abs_norm

    # rel_c = []
    # for attr in attribution.generate(sample, conditions, composite, record_layer=layer_names, batch_size=10):
    #     masked = attr.heatmap * mask[None, :, :]
    #     rel_c.append(torch.sum(masked, dim=(1, 2)))



    # Your tensor rel_value
    rel_value = rel_c.detach().cpu().numpy()

    # Calculate the maximum value in rel_value
    max_value = max(rel_value)  # Convert to tensor to use torch.max

    # Calculate the top 10% threshold
    threshold = 0.1 * max_value


    # Filter the tensor to get only the values that are greater than or equal to the threshold
    top_10_percent_rel = rel_value >= threshold


    top_10_percent_rel_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    # Find indices of True values
    indices_of_true = torch.nonzero(top_10_percent_rel_tensor, as_tuple=True)[0]

    # print("Indices of True values:", indices_of_true)

    # print("index of top 10 % : ",)
    # Convert the top 10% indices back to a tensor for further processing
    rel_value_cpu_tensor = torch.from_numpy(top_10_percent_rel).to(device)

    len_10_percent_indices = sum(top_10_percent_rel)

    print("layer 0 --len_10_percent_indices",len_10_percent_indices)
    print("layer 0 --10_percent_rel_indices",indices_of_true)
    print("layer 0 --10_percent_rel_data",abs_norm(rel_c)[top_10_percent_rel]*100 )



    # Convert the tensor to CPU for plotting
    # rel_value_cpu = rel_value.detach().cpu().numpy()
    rel_value_cpu = rel_value

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot all values
    # plt.plot(abs_norm(rel_value_cpu)*100, label='rel_value', marker='o', color='blue')
    plt.plot(abs_norm(torch.tensor(rel_value_cpu)), label='rel_value', marker='o', color='blue')

    # Highlight the top 10% values
    plt.plot(torch.arange(len(rel_value))[top_10_percent_rel].cpu(), 
            abs_norm(torch.tensor(rel_value_cpu))[top_10_percent_rel], 
            label='Top 10% rel_value', 
            marker='o', color='red', linestyle='None')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('rel_value')
    plt.title('rel_value Tensor with Top 10% Highlighted')
    plt.legend()

    # Show the plot
    # plt.show()
    relevance_plot
    save_path = '/home/prghosh/output_image/image_512/'
    plt.savefig(f'{save_path}{relevance_plot[12]}', format='jpg')
    # plt.savefig('/home/prghosh/output_image/image_512/rel_value_top_10_percent.jpg', format='jpg')  

    # Optionally, close the plot to free up resources
    plt.close()





    indices = torch.topk(rel_c, len_10_percent_indices).indices
    # we norm here, so that we clearly see the contribution inside the masked region as percentage
    print("layer 0 --indices of top 10% rel_c----: ",indices)
    print("layer 0 --top 10% rel_c with percentage----: ",abs_norm(rel_c)[indices]*100)




    import torch

    # Your tensor
    concept_ids = indices

    # Convert the tensor to a list
    concept_ids_list_12 = concept_ids.tolist()

    # Print the resulting list
    print("layer 0 --concept_ids_list : ",concept_ids_list_12)


    #################################################################################################







    heatmap_s=[]
    for i in concept_ids_list_12:
        conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': concept_ids_list_9,'features.7': concept_ids_list_10,'features.3': concept_ids_list_11,'features.0': [i], 'y': [208]}]
        heatmap, _, _, _ = attribution(sample, conditions, composite)
            # Convert heatmap to numpy array if it's not already in a compatible format
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        elif not isinstance(heatmap, np.ndarray):
            raise TypeError("Heatmap should be a torch.Tensor or np.ndarray")
        # print(conditions)
        heatmap_s.append(heatmap)
        
    # conditions = [{'features.40': [i], 'y': [208]} for i in concept_ids_list]
    # print(conditions)
    # heatmap, _, _, _ = attribution(sample, conditions, composite)


            


    # Convert the list of heatmaps to images without text
    heatmap_images = []
    for heatmap in heatmap_s:
        img = imgify(heatmap, symmetric=True)  # imgify should return a PIL image
        heatmap_images.append(img)

    # Create a grid image from the list of images
    def create_image_grid(images, grid_size):
        """Create a grid of images."""
        # Assuming images are PIL Images and have the same size
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        max_height = max(heights)
        
        grid_width = grid_size[0] * max_width
        grid_height = grid_size[1] * max_height
        
        new_image = Image.new('RGB', (grid_width, grid_height))
        
        x_offset = 0
        y_offset = 0
        
        for i, img in enumerate(images):
            new_image.paste(img, (x_offset, y_offset))
            x_offset += max_width
            if (i + 1) % grid_size[0] == 0:
                x_offset = 0
                y_offset += max_height
                
        return new_image

    # Determine grid size
    grid_size = (5,8)  # (rows, columns)
    final_image = create_image_grid(heatmap_images, grid_size)

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = file_name_new[12]
    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    # Display the resulting image
    # final_image.show()
    print(f"Saved heatmap grid as {file_name_new[12]}")



    print("--------------------------final heat map image------------------------------")


    conditions = [{'features.40': concept_ids_list ,'features.37': concept_ids_list_1,'features.34': concept_ids_list_2,'features.30': concept_ids_list_3,'features.27': concept_ids_list_4,'features.24': concept_ids_list_5,'features.20': concept_ids_list_6,'features.17': concept_ids_list_7,'features.14': concept_ids_list_8,'features.10': concept_ids_list_9,'features.7': concept_ids_list_10,'features.3': concept_ids_list_11,'features.0': concept_ids_list_12, 'y': [208]}]
    print("conditions for final heat map image: ",conditions)
    heatmap, _, _, _ = attribution(sample, conditions, composite)



        
    # Create the final image
    final_image = imgify(heatmap, symmetric=True)

    # Convert to 'RGB' mode if necessary
    if final_image.mode != 'RGB':
        final_image = final_image.convert('RGB')

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = "final_image_heat_map.jpg"

    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    print(f"Saved heatmap grid as {file_name}")


    print("------------------------------------------end--------------------------------------------")

    print("----------------------------------4---------------------------------------------------")
    print("--------------------------------complement-----------------------------------------------------")



    data1 = conditions[0]   # we are taking the 1st element of the list which is a dict
    # Total counts for each feature

    total_counts = {
    'features.40': 512, 'features.37': 512, 'features.34': 512, 'features.30': 512,
    'features.27': 512, 'features.24': 512, 'features.20': 256, 'features.17': 256,
    'features.14': 256, 'features.10': 128, 'features.7': 128, 'features.3': 64,
    'features.0': 64
    }

    # Function to compute complements
    def compute_complements(data1, total_counts):
        complements = {}
        for key, total in total_counts.items():
            existing_numbers = set(data1[key])
            full_range = set(range(total))
            complement = sorted(full_range - existing_numbers)
            complements[key] = complement
        return complements

    # Compute complements
    complements = compute_complements(data1, total_counts)
    complements['y'] = [208]

    conditions = [complements]




    # conditions = [{
    #                 'features.40': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 265, 266, 267, 268, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511],
    #                 'features.37': [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 65, 66, 67, 68, 69, 70, 71, 73, 74, 76, 77, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 120, 121, 122, 123, 125, 126, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200, 201, 202, 205, 206, 207, 208, 209, 211, 213, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 267, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 319, 320, 321, 322, 323, 326, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 360, 361, 362, 363, 364, 366, 367, 369, 370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 395, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 410, 411, 412, 413, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 427, 429, 430, 432, 433, 434, 436, 437, 438, 440, 441, 442, 443, 444, 446, 447, 448, 449, 451, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 491, 492, 493, 495, 496, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511],
    #                 'features.34': [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 157, 159, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 275, 276, 277, 278, 279, 280, 281, 282, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 342, 343, 345, 346, 347, 348, 349, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 407, 408, 409, 410, 411, 412, 413, 415, 416, 417, 419, 420, 421, 422, 423, 424, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 439, 440, 441, 442, 444, 445, 446, 447, 448, 449, 450, 452, 453, 454, 455, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 473, 474, 475, 476, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511], 
    #                 'features.30': [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 194, 195, 196, 197, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 402, 403, 404, 405, 406, 407, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 459, 460, 461, 462, 463, 464, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511], 
    #                 'features.27': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 39, 41, 42, 43, 44, 45, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 123, 124, 125, 126, 127, 128, 129, 130, 133, 134, 135, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 202, 203, 204, 205, 206, 207, 209, 210, 211, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 237, 238, 239, 240, 242, 243, 244, 245, 247, 248, 249, 250, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 308, 309, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 389, 390, 391, 392, 393, 394, 396, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 448, 450, 451, 452, 454, 455, 456, 457, 458, 459, 460, 461, 463, 464, 465, 466, 470, 471, 472, 473, 474, 475, 476, 477, 478, 480, 482, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511], 
    #                 'features.24': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 245, 246, 247, 248, 249, 250, 251, 252, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 364, 365, 366, 367, 368, 369, 371, 372, 373, 374, 375, 376, 377, 378, 380, 381, 382, 383, 384, 386, 387, 388, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 469, 470, 471, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511], 
    #                 'features.20': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 200, 201, 203, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 231, 233, 234, 236, 237, 238, 239, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255], 
    #                 'features.17': [0, 1, 3, 4, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19, 23, 25, 26, 27, 28, 30, 32, 33, 34, 35, 40, 42, 43, 45, 46, 47, 50, 52, 53, 54, 55, 56, 57, 59, 62, 63, 64, 66, 70, 72, 74, 78, 79, 80, 82, 86, 87, 88, 89, 91, 92, 96, 97, 98, 99, 102, 103, 105, 107, 108, 110, 112, 114, 117, 118, 120, 122, 123, 124, 125, 126, 127, 128, 130, 133, 134, 135, 137, 138, 141, 147, 150, 151, 153, 154, 155, 156, 157, 161, 164, 165, 167, 169, 170, 171, 173, 174, 176, 184, 187, 188, 190, 191, 192, 196, 197, 199, 201, 202, 203, 205, 206, 209, 210, 211, 214, 216, 217, 218, 219, 220, 222, 223, 228, 229, 230, 232, 233, 234, 235, 237, 239, 240, 242, 244, 245, 246, 247, 249, 252, 253, 254], 
    #                'features.14': [6, 8, 9, 14, 17, 18, 25, 26, 27, 28, 29, 37, 38, 42, 44, 45, 47, 48, 52, 54, 55, 59, 68, 72, 73, 74, 76, 77, 81, 82, 83, 86, 89, 90, 92, 93, 94, 106, 108, 109, 110, 111, 112, 113, 116, 119, 123, 124, 125, 126, 129, 130, 133, 134, 137, 140, 141, 142, 143, 148, 149, 152, 156, 157, 162, 163, 168, 170, 173, 176, 181, 189, 190, 197, 198, 200, 202, 203, 204, 207, 211, 214, 217, 220, 221, 228, 234, 237, 239, 240, 242, 245, 248, 251, 254, 255],
    #                'features.10': [0, 8, 16, 57, 58, 80, 89, 105, 119, 123], 
    #                'features.7': [2, 6, 9, 10, 11, 13, 14, 15, 20, 24, 25, 26, 28, 29, 30, 38, 40, 42, 43, 45, 48, 49, 51, 52, 53, 55, 66, 71, 75, 76, 77, 80, 84, 85, 86, 92, 93, 102, 106, 108, 111, 115, 122], 
    #                'features.3': [0, 12, 17, 30, 34, 36, 43, 44, 46, 50, 52, 55, 57, 61], 
    #                 'features.0': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 19, 20, 21, 22, 24, 25, 26, 29, 31, 32, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], 
    #                 'y': [208]}]

    print("conditions for complement of final heat map image: ",conditions)
    heatmap, _, _, _ = attribution(sample, conditions, composite)



        
    # Create the final image
    final_image = imgify(heatmap, symmetric=True)

    # Convert to 'RGB' mode if necessary
    if final_image.mode != 'RGB':
        final_image = final_image.convert('RGB')

    # Specify the path and filename
    save_directory = "/home/prghosh/output_image/image_512/"  # Update this to your desired directory

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = "complement_final_image_heat_map.jpg"

    file_path = os.path.join(save_directory, file_name)

    # Save the image as a JPG file
    final_image.save(file_path, format="JPEG")

    print(f"Saved heatmap grid as {file_name}")
sys.stdout = sys.__stdout__

print("Output has been saved to the file.")






# ['features.0',
#  'features.3',
#  'features.7',
#  'features.10',
#  'features.14',
#  'features.17',
#  'features.20',
#  'features.24',
#  'features.27',
#  'features.30',
#  'features.34',
#  'features.37',
#  'features.40',
#  'classifier.0',
#  'classifier.3',
#  'classifier.6']

