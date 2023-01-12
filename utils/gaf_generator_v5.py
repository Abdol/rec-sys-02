# Imports
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from typing import *
from multiprocessing import Pool
import datetime as dt
matplotlib.use('Agg')

# Constants
export_folder_name = "gaf"
dataset_folder_name = '30_11_2022'
training_output_dataset_filename = 'gaf_dataset_training.csv'
test_output_dataset_filename = 'gaf_dataset_test.csv'
GAF_RESOLUTION = 128
IMAGE_SIZE= (5,5) # in inches
frequency = '1h'
parameter_name = 'state'
col_name = ['ts', 'datetime','state']
convert_dict = {'ts': int,
                'state': int}

files_list = [
    'living_room_tv', 
    'kitchen_kettle',
    'office_computer_setup_01',
    'kitchen_toaster',
    'kitchen_washing_machine',
    'office_computer_setup_02',
    'kitchen_fridge',
    'living_room_humidity',
    'living_room_temperature',
    'office_humidity',
    'office_temperature',
    'kitchen_occupancy',
    'living_room_occupancy',
    ]

def import_data(dataset_path, cols):
    df = pd.read_csv(dataset_path, names=cols, header=None)
    convert_dict = {'ts': int,
                    'state': float}
    df = df.drop([0]) # Remove first header row
    # df = df.head(1000) # remove at production
    df = df.astype(convert_dict)
    df['datetime'] = df['ts'].apply(lambda x: datetime.utcfromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
    df['DateTime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
    df = df.drop(columns=['ts', 'datetime'])
    print(df.head())
    return df

def create_gaf(ts) -> Dict[str, Any]:
    """
    :param ts:
    :return:
    """
    data = dict()
    gasf = GramianAngularField(method='summation', image_size=ts.shape[0])
    data['gasf'] = gasf.fit_transform(pd.DataFrame(ts).T)[0]
    data['gasf_mean'] = np.mean(np.mean(data['gasf'], axis=0), axis=0)
    return data

def create_images(X_plots: Any, image_name: str, output_path: str, destination: str, image_matrix: tuple =(1, 1), mix_classes: bool = False) -> None:
    # output_dataset_image_list.append(image_name)
    fig = plt.figure(figsize=IMAGE_SIZE, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    images = X_plots
    ax.imshow(images[0], cmap='rainbow', origin='lower')

    repo = os.path.join(output_path, destination) if mix_classes == False else output_path
    fig.savefig(os.path.join(repo, image_name), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return {'image_name': image_name+'.png', 'label': destination}


def main():

    for filename in files_list:
        print('Processing', filename)
        _dataset_path = dataset_folder_name + '/' + filename
        df = import_data(_dataset_path + '.csv', col_name)
        destination_folder = filename
        PATH = os.path.abspath('')
        GAF_PATH = os.path.join(PATH , export_folder_name)
        TRAINING_OUTPUT_PATH = os.path.join(GAF_PATH , destination_folder)
        os.makedirs(TRAINING_OUTPUT_PATH, exist_ok=True)
        
        df = df.groupby(pd.Grouper(key='DateTime', freq=frequency)).mean().interpolate().reset_index() 
        list_dates = df['DateTime'].apply(str).tolist() 
        index = 0
        box_size = GAF_RESOLUTION
        decision_map = {key: [] for key in [destination_folder]} # Container to store data_slice for the creation of GAF
        while True:
            if index >= len(list_dates) - 2:
                break
            # Select appropriate timeframe
            data_slice = df.loc[(df['DateTime'] < list_dates[len(list_dates) - 1]) & (df['DateTime'] > list_dates[index])]
            # print("DATA SLICE==========================================", index)
            # print(data_slice)
            # print("DATA SLICE END======================================")
            gafs = []
            # Group data_slice by period
            for freq in [frequency]:
                group_dt = data_slice.groupby(pd.Grouper(key='DateTime', freq=freq)).mean().reset_index()
                group_dt = group_dt.dropna()
                gafs.append(group_dt[parameter_name].head(box_size))
            decision_map[destination_folder].append([list_dates[index], gafs])
            index += 1
        # Generate the images from processed data_slice
        print('Starting image generation...')
        output_dataset = pd.DataFrame(columns = ['image_name','label'])
        main_decision = list(decision_map.keys())[0]
        for decision, data in decision_map.items():
            for i, image_data in enumerate(data):
                # print('decision', decision)
                # print('image_data', image_data)
                to_plot = [create_gaf(x)['gasf'] for x in image_data[1]]
                gaf_mean = [create_gaf(x)['gasf_mean'] for x in image_data[1]]
                i = len(output_dataset)
                output_path = TRAINING_OUTPUT_PATH
                mix_classes = True
                output_dataset.loc[i] = create_images(X_plots=to_plot,
                                image_name='{0}'.format(image_data[0].replace('-', '_')),
                                output_path=output_path, destination=destination_folder, mix_classes=mix_classes)
                
        # Convert output_dataset to csv
        dataset_filename = GAF_PATH + '/' + filename + '_' + training_output_dataset_filename 
        output_dataset.to_csv(dataset_filename, index=False)
        total_images = len(decision_map[destination_folder])
        images_created = total_images
        print("========GAF REPORT========:\nTotal Images Created: {0}".format(images_created))

main()
