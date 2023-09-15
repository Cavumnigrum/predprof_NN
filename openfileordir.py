import tkinter as tk
import tkinter.filedialog as fd
from PIL import Image, ImageTk
import cv2
from imageio import imread
import os
import matplotlib.pyplot as plt
from IPython.display import display
from tensorflow.keras.models import Model
import config as cfg
import pickle
import tkinter.messagebox as mb

from config import nn_base, rpn_layer, classifier_layer, format_img_size, format_img, format_img_channels, get_real_coordinates, rpn_to_roi, non_max_suppression_fast,Config
import numpy as np
from tensorflow.keras.layers import Input
from keras import backend as K

# WYSI 727 WYSI 727 WYSI 727 WYSI 727 WYSI 727
a = []
ss = []
fdir = []
xxx = []
fig,ax = plt.subplots()
s = 0

config_output_filename = 'model_vgg_config.pickle'

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)


C.model_path = 'model_frcnn_vgg.hdf5'
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
print(C.model_path)
bbox_threshold = 0.7
n =0

num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_layers = nn_base(img_input, trainable=True)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)

classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


def calculate():
    print(fdir)
    total_ears = 0
    all_dets = []
    for idx, imagepath in enumerate(fdir):
        img = cv2.imread(imagepath)
        X, ratio = format_img(img, C)

        X = np.transpose(X, (0, 2, 3, 1))

        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes = {}
        probs = {}
        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
        all_dets = []
        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                all_dets.append((key,100*new_probs[jk]))

        total_ears+=len(all_dets)
        print(all_dets, len(all_dets),total_ears)
        global s
        if s!=0:
            msg = 'Количество колосков на последней фотографии равно {}, на всех фотографиях - {} \n Средняя площадь поля равна {}'.format(len(all_dets),total_ears,total_ears/s)
        else:
            msg = 'Количество колосков на последней фотографии равно {}, на всех фотографиях - {} \n Средняя площадь не определена, т.к. не введены границы поля'.format(len(all_dets),total_ears)
        mb.showinfo('Количество колосков', msg)

def clear():
    global a,ss,fdir,xxx
    a = []
    ss = []
    fdir = []
    xxx = []

def get_this_shit():
    global new_entry
    blup = input.get()
    lbl_result["text"] = f"{blup}"

    input.delete(0,tk.END)
def get_new_entry():
    n = new_entry.get()
    a.append(n)
    new_entry.delete(0,tk.END)
def plot_show():
    global s
    for i in a:
        i = i.strip('(){}[]')
        i = i.split(',')
        xxx.append(int(i[0]))
        ss.append(int(i[1]))
    for i in range(len(a)):
        s+= xxx[i]*ss[(i+1)%len(ss)]-xxx[(i+1)%len(xxx)]*ss[i]
    s = abs(s/2)
    ax.fill(xxx,ss)
    ax.set_facecolor('seashell')
    ax.set_title('Площадь этого поля равна {}'.format(s))
    fig.set_figwidth(6)
    fig.set_figheight(3)
    fig.set_facecolor('floralwhite')
    plt.savefig('field.png')
    image = Image.open('field.png')
    canvas = tk.Canvas(height=image.size[1], width=image.size[0])
    photo = ImageTk.PhotoImage(image)
    image = canvas.create_image(0, 0, anchor='nw',image=photo)
    canvas.grid(row=0,column=2)

    show()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('1564-7')
        master_frame = tk.Frame(master = self, height = 100, width = 100)
        master_frame.grid()

        btn_file = tk.Button(master = master_frame,text = 'Выбрать файл',
                            command = self.choose_file)

        btn_dir = tk.Button(master = master_frame,text = 'Выбрать папку',
                            command=self.choose_directory)
        label_ver = tk.Label(text = 'Введите координаты вершин *пример:(X,Y)*')

        label_ver.grid(row = 1, column = 0)

        btn_file.grid(row = 4,column = 0)

        btn_dir.grid(row = 4,column = 1)
    # def choose_file(self):
    #     filetypes = (('Изображение','*.png *.jpg'),('Любой','*'))
    #     filename = fd.askopenfilename(title = 'Открыть файл', initialdir = '/',
    #                                 filetypes = filetypes)
    #     if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tiff'):
    #         print(filename,isinstance(filename,str))
    #         fdir.append(filename)
    #         print(fdir,'dsa')
    #         image = Image.open(filename)
    #         caav = tk.Canvas(height=image.size[1]//2, width=image.size[0]//2)
    #         photo = ImageTk.PhotoImage(image)
    #         image = caav.create_image(0, 0, anchor='nw',image=photo)
    #         caav.grid(row=3,column=0)
    #         show()
    # def choose_directory(self):
    #     directory = fd.askdirectory(title = 'Открыть папку', initialdir = '/')
    #     print(directory)
    #     print(os.listdir(path = directory))
    #     if directory:
    #         for i in os.listdir(path = directory):
    #             if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.tiff'):
    #                 fdir.append(directory+'/'+i)
if __name__ == '__main__':
    app = App()
    frame = tk.Frame(master = app)


    btn_go = tk.Button(master = app,text = "\N{RIGHTWARDS BLACK ARROW}",command = calculate)
    lbl_result = tk.Label(master = app,text = '!!!')

    frame.grid(row = 1, column = 1, padx = 10)
    btn_go.grid(row = 1,column = 3, padx = 10)
    lbl_result.grid(row = 1,column = 4, padx = 10)

    label_cord = tk.Label(master = app, text = '')
    label_cord.grid(row = 2,column = 3)

    new_entry = tk.Entry(master = frame)
    new_entry.grid(row = 5, column = 4)

    btn_one = tk.Button(master = frame, text= 'Ввод', command = get_new_entry)
    btn_two = tk.Button(master = frame,text = 'Визуализировать поле', command = plot_show)
    btn_clear = tk.Button(master = app, text = 'Очистить входные данные', command = clear)

    btn_one.grid(row = 5, column = 0)
    btn_two.grid(row = 5, column = 1)
    btn_clear.grid(row = 10, column = 0)
    app.mainloop()
