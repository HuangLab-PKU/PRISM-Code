import warnings
warnings.filterwarnings("ignore")

import os 
from pathlib import Path
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


global window
global workdir
global G_layer
global color_grade

workdir = Path(__file__)
main_picture = Path(__file__).parent/'lib'/'main_page.png'

G_layer = 2
color_grade = 5
num_per_layer = (color_grade + 1) * color_grade // 2


# Define the layout of the GUI
overview_layout = [
    [sg.Text("Step 1: Select parameters")],
    [sg.Text('Color_Grade:'), sg.Input(key='Color_Grade', default_text=5)],    
    [sg.Text('Ref_channel_layer:'), sg.Input(key='Ref_channel_layer', default_text=2)],

    [sg.Text("Step 1: Upload Files")],
    [sg.Text('Select a workdir:'), sg.Input(key='wkdir'), sg.FolderBrowse()],
    [sg.Text('Select a file to upload:'), sg.Input(key='file'), sg.FileBrowse(), sg.Button('Upload', key='upload_csv')],

    [sg.Button("Data_overview", key='overview_hist'), sg.Button("save", key='overview_hist_save')],
    [sg.Image(filename=main_picture, key='overview_hist_canvas'),],
]


from lib.data_preprocess import overview
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk


def dir_initialization(workdir=workdir):
    os.makedirs(workdir, exist_ok=True)
    figure_dir = os.path.join(workdir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    return workdir, figure_dir


# from lib.preprocess import 
parameter_layout = [
    [sg.Text("Adjust Gaussian and Parameters")],
    [sg.Text("Gaussian and kde accuracy Parameters should be adjusted until you get expected number of peaks. This step is done to get better initial centroid points for better cluster results.")],
    [
        sg.Text("Gaussian"),
    ],
    [
        sg.Text("Gau_0"),
        sg.Slider(
            range=(0, 0.2),
            resolution=0.01,
            default_value=0.03,
            orientation="h",
            size=(20, 15),
            key="gau_0",
        ),
        sg.Text("Gau_1"),
        sg.Slider(
            range=(0, 0.2),
            resolution=0.01,
            default_value=0.03,
            orientation="h",
            size=(20, 15),
            key="gau_1",
        ),
        sg.Text("Gau_ref"),
        sg.Slider(
            range=(0, 0.2),
            resolution=0.01,
            default_value=0.03,
            orientation="h",
            size=(20, 15),
            key="gau_ref",
        ),
    ],
    
    [
        sg.Text("G_kde"),
        sg.Slider(
            range=(0, 2),
            resolution=0.01,
            default_value=1,
            orientation="h",
            size=(20, 15),
            key="G_kde",
        ),
        sg.Text("Y_kde"),
        sg.Slider(
            range=(0, 2),
            resolution=0.01,
            default_value=1,
            orientation="h",
            size=(20, 15),
            key="Y_kde",
        ),
        sg.Text("B_kde"),
        sg.Slider(
            range=(0, 2),
            resolution=0.01,
            default_value=1,
            orientation="h",
            size=(20, 15),
            key="B_kde",
        ),
        sg.Text("R_kde"),
        sg.Slider(
            range=(0, 2),
            resolution=0.01,
            default_value=1,
            orientation="h",
            size=(20, 15),
            key="R_kde",
        ),
        sg.Button('submit', key='total_submit')
    ],
    
    [
        sg.Text("Y_kde_layer"),
        sg.Slider(
            range=(0, 2),
            resolution=0.01,
            default_value=1,
            orientation="h",
            size=(20, 15),
            key="Y_kde_layer",
        ),
        sg.Text("B_kde_layer"),
        sg.Slider(
            range=(0, 2),
            resolution=0.01,
            default_value=1,
            orientation="h",
            size=(20, 15),
            key="B_kde_layer",
        ),
        sg.Text("R_kde_layer"),
        sg.Slider(
            range=(0, 2),
            resolution=0.01,
            default_value=1,
            orientation="h",
            size=(20, 15),
            key="R_kde_layer",
        ),
        sg.Button('submit', key='layer_submit'),
    ],

    [
        sg.Image(filename=main_picture, key='adjustment_hist_canvas'), 
        sg.Image(filename=main_picture, key='adjustment_hist_by_layer_canvas'),
    ]
]


GMM_layout = [
    [sg.Button("Perform GMM"), sg.Text('It may take minutes, please wait.')],
    [sg.Text('Results will be shown below:')],
    [sg.Image(filename=main_picture, key='GMM_result_canvas'),],
    [sg.Image(filename=main_picture, key='count_GMM'),],
]


import io
import PySimpleGUI as sg
from PIL import Image, ImageDraw, ImageTk

# Function to convert the PIL image to a format that PySimpleGUI can display
def convert_to_bytes(image):
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()

# Load the initial image and create a drawing object
original_image = Image.open(main_picture)
canvas_size = original_image.size

# Create a separate mask image (initially transparent)
mask = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
draw = ImageDraw.Draw(mask)
layer_list = [f'layer{layer}' for layer in range(G_layer)]

# Define the layout for the drawing tab
drawing_tab_layout = [
    [sg.Text('Select Layer:'), sg.Combo(layer_list, default_value=layer_list[0], key='-LAYER-', enable_events=True)],
    [sg.Canvas(size=canvas_size, key='-CANVAS-')],
    [sg.Text('Cluseter'), sg.Slider(range=(1, 15), orientation='h',  key='-cluster-'), 
     sg.Button('Save', key='-SAVE-'), 
     sg.Button('Clear', key='-CLEAR-'), 
     sg.Button('Remove_saved_masks', key='-REMOVEMASK-'),],
    [sg.Text('Relabel')], 
    [sg.Text('Mode'), sg.Combo(['discard', 'replace'], default_value='discard', key='-RELAYER_MODE-'), sg.Button('Perform', key='-DONE-'), sg.Button('Reset', key='-RESET-'), ],
    [sg.Button('Show_result', key='-VISUALIZE-')],
    [sg.Image(filename=main_picture, key='relabel_result_canvas'),],
]


evaluation_layout = [
    [sg.Button("Evaluation", key='-SHOW_EV-'), sg.Checkbox(text='Mannual_Thre', key='MannualThre')],
    [sg.Image(filename=main_picture, key='conut_distribution_canvas'),],
    [sg.Image(filename=main_picture, key='cdf_heatmap_canvas'),],
    [sg.Button("Save", key='-SAVE_DF-'), sg.Input(default_text=os.path.join(workdir, 'intensity_labeled.csv'), key='save_path')],
]


# Create the TabGroup element
tab_titles = [
    "Data_overview", 
    "Gaussian_adjustment",
    "GMM_and_results",
    "Mannual_select",
    "Evaluation",
    ]

tab_group = sg.TabGroup(
    [
        [
            sg.Tab(tab_titles[0], overview_layout),
            sg.Tab(tab_titles[1], parameter_layout),
            sg.Tab(tab_titles[2], GMM_layout),
            sg.Tab(tab_titles[3], drawing_tab_layout),
            sg.Tab(tab_titles[4], evaluation_layout),
        ], 
    ],
    key='-TABGROUP-'
)

layout = [[tab_group], [sg.Button("Next", key='-NEXT-'), sg.Button('Exit', key='-EXIT-'), ]]
current_tab_index = 0


# Create the window
window = sg.Window('Multi-Tab Application', layout, finalize=True)
canvas = window['-CANVAS-'].TKCanvas

intensity_file = None
intensity_thre = None
workdir, figure_dir = dir_initialization('./')


global RYB_x_transform, RYB_y_transform, RYB_xy_transform
RYB_x_transform = np.array([[-np.sqrt(2) / 2], [0], [np.sqrt(2) / 2]])
RYB_y_transform = np.array([[-np.sqrt(3) / 3], [2 / np.sqrt(3)], [-np.sqrt(3) / 3]])
RYB_xy_transform = np.concatenate([RYB_x_transform, RYB_y_transform], axis=1)


# Function to update the canvas with the blended image
def update_canvas(original_image, mask):
    global tk_image
    canvas.delete("all")  # Clear the current image
    # Blend the mask with the original image
    blended_image = Image.alpha_composite(original_image.convert("RGBA"), mask)
    tk_image = ImageTk.PhotoImage(blended_image)
    canvas.create_image(0, 0, image=tk_image, anchor='nw')

update_canvas(original_image, mask)

# Function to handle drawing
def draw_circle(event):
    x, y = event.x, event.y
    radius = 5  # Adjust radius as needed
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 0, 0, 128))  # Semi-transparent red
    update_canvas(original_image, mask)
    
# Bind mouse events to the drawing function
canvas.bind('<B1-Motion>', draw_circle)  # Draw on drag


# Event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == '-EXIT-':
        break

    if event == '-NEXT-':
        current_tab_index = (current_tab_index + 1) % len(tab_titles)
        window['-TABGROUP-'].Widget.select(current_tab_index)  # Change the selected tab

    ## init tab
    active_tab = values['-TABGROUP-']
    if active_tab == tab_titles[0]:
        current_tab_index = 0
        G_layer = int(values['Ref_channel_layer'])
        color_grade = int(values['Color_Grade'])
        num_per_layer = (color_grade + 1) * color_grade // 2

        layer_list = [f'layer{layer+1}' for layer in range(G_layer)]
        window['-LAYER-'].update(values=layer_list)

        if values['wkdir'] != '':
            workdir = values['wkdir']
            window['-SAVE_DF-'].update(os.path.join(workdir, 'intensity_labeled.csv'))
        workdir, figure_dir = dir_initialization(workdir)

        if event == 'upload_csv':
            src_path = values['file']
            try:
                intensity_file = pd.read_csv(src_path, index_col=False)
                sg.popup('File uploaded successfully!')
            except:
                sg.popup('Please select valid file path!')
        
        if event == "overview_hist":
            if intensity_file is not None:
                out_path = os.path.join(figure_dir, 'overview_hist.jpg')
                overview(intensity_file, sample=10000, save=True, save_quality='low', out_path=out_path)
                pil_image = Image.open(out_path)
                photo_img = ImageTk.PhotoImage(pil_image)
                window['overview_hist_canvas'].update(data=photo_img)
            else:
                sg.popup('Please upload file first!')
        
        if event == 'save':
            if intensity_file is not None:
                out_path = os.path.join(figure_dir, 'overview_hist.jpg')
                overview(intensity_file, sample=False, save=True, save_quality='high', out_path=out_path)
                sg.popup('Successfully saved overview histplot!')
            else:
                sg.popup('Please upload file first!')

    
    ## parameter tab
    if active_tab == tab_titles[1]:
        current_tab_index = 1
        if event == 'total_submit':
            if intensity_file is not None:
                from lib.data_preprocess import gau_hist
                from lib.data_preprocess import gaussian_blur
                gau_0 = values['gau_0']
                gau_1 = values['gau_1']
                gau_ref = values['gau_ref']
                intensity = gaussian_blur(intensity=intensity_file,
                                               RYB_x_transform=RYB_x_transform,
                                               RYB_y_transform=RYB_y_transform,    
                                               gau_0=gau_0,
                                               gau_1=gau_1,
                                               gau_ref=gau_ref,
                                               )
                out_path = os.path.join(figure_dir, 'gau_hist.png')
                G_kde, Y_kde, B_kde, R_kde = values['G_kde'], values['Y_kde'], values['B_kde'], values['R_kde']
                Y_maxima, B_maxima, R_maxima, G_minima, intensity = gau_hist(
                    intensity_fra=intensity,
                    G_layer=G_layer,
                    color_grade=color_grade,
                    G_kde=G_kde, Y_kde=Y_kde, B_kde=B_kde, R_kde=R_kde,
                    out_path=out_path,
                    )
                pil_image = Image.open(out_path)
                photo_img = ImageTk.PhotoImage(pil_image)
                window['adjustment_hist_canvas'].update(data=photo_img)
            else:
                sg.popup('Please upload file first!')

        if event == 'layer_submit':
            from lib.data_preprocess import gau_hist_by_layer
            out_path = os.path.join(figure_dir, 'layer_hist.png')
            Y_kde, B_kde, R_kde = values['Y_kde_layer'], values['B_kde_layer'], values['R_kde_layer']
            centroid_init_dict = gau_hist_by_layer(intensity_fra=intensity, 
                            G_layer=G_layer, 
                            color_grade=color_grade,
                            R_maxima=R_maxima,
                            Y_maxima=Y_maxima,
                            B_maxima=B_maxima,
                            Y_kde=Y_kde, B_kde=B_kde, R_kde=R_kde,
                            out_path=out_path,
                            )
            pil_image = Image.open(out_path)
            photo_img = ImageTk.PhotoImage(pil_image)
            window['adjustment_hist_by_layer_canvas'].update(data=photo_img)
            

    ## GMM_tab
    if active_tab == tab_titles[2]:
        current_tab_index = 2
        if event == 'Perform GMM':
            from lib.GMM_and_visualization import GMM_by_layer, GMM_visualization
            from lib.quantitative_evaluation import conut_distribution
            intensity, GMM_dict = GMM_by_layer(
                intensity,
                G_layer,
                num_per_layer,
                channel_to_use=["Ye/A", "B/A", "R/A"],
                centroid_init_dict=centroid_init_dict,
                )
            GMM_visualization(
                intensity_fra=intensity,
                G_layer=G_layer,
                num_per_layer=num_per_layer,
                GMM_dict=GMM_dict,
                centroid_init_dict=centroid_init_dict,
                RYB_xy_transform=RYB_xy_transform,
                out_path_dir=figure_dir,)
            pil_image = Image.open(os.path.join(figure_dir, "mousebrain_scatter_GMM_cluster_by_layer.jpg"))
            photo_img = ImageTk.PhotoImage(pil_image)
            window['GMM_result_canvas'].update(data=photo_img)
            
            conut_distribution(intensity_fra=intensity,
                               num_per_layer=num_per_layer,
                               G_layer=G_layer,
                               out_path=os.path.join(figure_dir,'count_GMM.jpg'),
                               )
            pil_image = Image.open(os.path.join(figure_dir, "count_GMM.jpg"))
            photo_img = ImageTk.PhotoImage(pil_image)
            window['count_GMM'].update(data=photo_img)


    ## draw to mannually circle tab
    if active_tab == tab_titles[3]:
        current_tab_index = 3
        cluster = int(values['-cluster-'])
        if event == '-LAYER-':
            selected_layer = values['-LAYER-']
            layer = int(selected_layer.replace('layer', ''))
            window['-cluster-'].Update(range=((layer-1) * num_per_layer + 1, layer * num_per_layer))
            try:
                original_image = Image.open(os.path.join(figure_dir, f'layer{layer}.jpg'))
            except:
                sg.popup('No valid layer to select, please run GMM first.')
            canvas_size = original_image.size
            mask = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(mask)
            update_canvas(original_image, mask)

        if event == '-CLEAR-':
            # Reset the mask to be fully transparent
            mask = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(mask)
            update_canvas(original_image, mask)

        if event == '-SAVE-':
            mask_dir = os.path.join(workdir,'masks')
            os.makedirs(mask_dir, exist_ok=True)
            mask.save(os.path.join(workdir, f'masks/mask_{cluster}.png'))  # Replace with desired save path for mask
        
        if event == '-REMOVEMASK-':
            mask_dir = os.path.join(workdir,'masks')
            os.makedirs(mask_dir, exist_ok=True)
            for filename in os.listdir(mask_dir):
                if filename.endswith('.png'):
                    os.remove(os.path.join(mask_dir, filename))

        if event == '-RESET-':
            intensity_thre = None

        if event == '-DONE-':
            from lib.manual_thre import relabel
            mode = values['-RELAYER_MODE-']
            if intensity_thre is not None:
                intensity_thre = relabel(intensity_fra=intensity_thre, mask_dir=os.path.join(workdir, 'masks'), mode=mode, num_per_layer=num_per_layer)

            elif intensity is not None:
                intensity_thre = relabel(intensity_fra=intensity, mask_dir=os.path.join(workdir, 'masks'), mode=mode, num_per_layer=num_per_layer)

            else:
                sg.popup('Intensity file not found, please run previous steps first.')

        if event == '-VISUALIZE-':
            if intensity_thre is not None:
                from lib.GMM_and_visualization import visualization
                visualization(
                    intensity_fra=intensity_thre, 
                    G_layer=G_layer,
                    num_per_layer=num_per_layer,
                    # RYB_xy_transform=RYB_xy_transform,
                    out_path_dir=os.path.join(figure_dir, "mousebrain_scatter_relabel_cluster_by_layer.jpg"))
                
                pil_image = Image.open(os.path.join(figure_dir, "mousebrain_scatter_relabel_cluster_by_layer.jpg"))
                photo_img = ImageTk.PhotoImage(pil_image)
                window['relabel_result_canvas'].update(data=photo_img)

            else:
                sg.popup('No relabel found, please see the scatter result at the previous page.')


    ## Cluster evaluation
    if active_tab == tab_titles[4]:
        current_tab_index = 4
        if event == '-SHOW_EV-':
            from lib.quantitative_evaluation import calculate_cdf, conut_distribution, cdf_heatmap

            mannual_thre = values['MannualThre']
            if mannual_thre:
                data = intensity_thre.copy()
            else:
                data = intensity.copy()

            conut_distribution(intensity_fra=data,
                               num_per_layer=num_per_layer,
                               G_layer=G_layer,
                               out_path=os.path.join(figure_dir,'count_final.jpg'),
                               )
            window['conut_distribution_canvas'].update(data=ImageTk.PhotoImage(Image.open(os.path.join(figure_dir, 'count_final.jpg'))))

            CDF_dict = dict()
            for layer in range(G_layer):
                CDF_dict[layer]= calculate_cdf(data, st=layer * num_per_layer, num_per_layer=num_per_layer)
            
            cdf_heatmap(
            intensity_fra=data,
            CDF_dict=CDF_dict,
            p_thre_list=[0.0001, 0.001, 0.01, 0.1],
            corr_method="spearman",
            out_path=os.path.join(figure_dir, 'cdf_heatmap.jpg'),
            G_layer=G_layer,
            num_per_layer=num_per_layer,
            )
            window['cdf_heatmap_canvas'].update(data=ImageTk.PhotoImage(Image.open(os.path.join(figure_dir, 'cdf_heatmap.jpg'))))
        
        if event == '-SAVE_DF-':
            mannual_thre = values['MannualThre']
            out_path = values['save_path']
            if mannual_thre:
                intensity_thre.to_csv(out_path)
            else:
                intensity.to_csv(out_path)

window.close()