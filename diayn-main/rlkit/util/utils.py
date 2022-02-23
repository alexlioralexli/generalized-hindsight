import numpy as np
import ffmpy, cv2, os
import os.path as osp

def int_to_onehot(x, dim):
    result = np.zeros([len(x), dim])
    result[np.arange(len(x)), x.flatten()] = 1.0
    return result

def save_video(rgb_array, save_path, text):

    if os.path.exists(save_path):
        os.remove(save_path)
    split_path = osp.splitext(save_path)
    temp_save_path = split_path[0] + "_temp" + split_path[1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(temp_save_path, fourcc, 24, (500, 500)) #todo: determine shape from rgb_array
    for i, I in enumerate(rgb_array):
        if i % 3 == 0:
            I = cv2.putText(I, text, (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA, True)
            I = cv2.flip(I[..., ::-1], 0)
            writer.write(I)
    writer.release()
    ff = ffmpy.FFmpeg(inputs={temp_save_path: None},
                      outputs={save_path:'-vcodec libx264 -crf 20'})
    ff.run()
    os.remove(temp_save_path)