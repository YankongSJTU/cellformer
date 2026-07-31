"""
UNet + Attention-based nuclear semantic segmentation module.

Runs TF1.x multi-scale UNet inference on 384x384 patches with overlap voting.
Executed via subprocess with the correct glibc loader to avoid compatibility issues.
"""

import cv2
import os
import sys
import shutil
import numpy as np


def _get_tf_python_cmd():
    """Get the correct Python command with glibc loader from bashrc alias."""
    loader = '/export/home/kongyan/local/glibc-2.42/lib/ld-linux-x86-64.so.2'
    lib_path = '/export/home/kongyan/local/glibc-2.42/lib:/export/home/kongyan/miniconda3/lib:/export/usr/cuda/cuda-12.2/lib64:/usr/lib64'
    python_bin = '/export/home/kongyan/miniconda3/bin/python'
    return [loader, '--library-path', lib_path, python_bin]


def _get_tf_python_str():
    """Return the shell-escaped string form of the loader+python command (for embedding in scripts)."""
    loader = '/export/home/kongyan/local/glibc-2.42/lib/ld-linux-x86-64.so.2'
    lib_path = '/export/home/kongyan/local/glibc-2.42/lib:/export/home/kongyan/miniconda3/lib:/export/usr/cuda/cuda-12.2/lib64:/usr/lib64'
    python_bin = '/export/home/kongyan/miniconda3/bin/python'
    return repr([loader, '--library-path', lib_path, python_bin])


def _generate_unet_script(image_paths, work_dir, gpu_id):
    """
    Generate a standalone Python script that runs the UNet segmentation.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modules_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(project_root, 'checkpoints', 'nucseg_unet')
    pred_dir = os.path.join(work_dir, 'unet_pred')
    tf_python_cmd_str = _get_tf_python_str()

    image_list_path = os.path.join(work_dir, 'image_list.txt')
    with open(image_list_path, 'w') as f:
        for p in image_paths:
            f.write(p + '\n')

    script_content = r'''#!/usr/bin/env python
import sys, os, shutil, subprocess as _sp

# The correct python command with glibc loader (needed for this system)
_TF_PYTHON_CMD = %(tf_python_cmd_str)s

# Ensure numpy<2.0 for TensorFlow C ABI compatibility
import numpy as _np_check
if _np_check.__version__.startswith('2.'):
    print("Downgrading numpy for TF compatibility (", _np_check.__version__, "-> 1.24.3)...")
    _sp.check_call(_TF_PYTHON_CMD + ['-m', 'pip', 'install', 'numpy==1.24.3', '--quiet', '--no-warn-conflicts'])
    print("Restarting script with downgraded numpy...")
    os.execv(_TF_PYTHON_CMD[0], _TF_PYTHON_CMD + sys.argv)  # re-exec with new numpy

import numpy as np
import cv2

# Patch numpy for old TensorFlow compatibility
_np_compat = {
    'bool8': np.bool_, 'int0': np.intp, 'uint0': np.uintp,
    'object0': np.object_, 'str0': np.str_, 'bytes0': np.bytes_,
    'bool': np.bool_, 'int': int, 'float': float, 'complex': complex,
    'object': np.object_, 'str': np.str_, 'bytes': np.bytes_,
}
for _a, _v in _np_compat.items():
    if not hasattr(np, _a):
        setattr(np, _a, _v)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = str(%(gpu_id)d)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, %(modules_dir)r)

import BatchDatsetReader as DataSetReader
import read_MITSceneParsingData as fashion_parsing
import TensorflowUtils as Utils
import PredFunc as fd
import resnet_modified as resnet

batch_size = 1
logs_dir = %(logs_dir)r
data_dir = %(pred_dir)r
IMAGE_SIZE = 384
NUM_OF_CLASSES = 2
learning_rate = 1e-4

def u_net_inference(image, is_training=False):
    net = {}
    l2_reg = learning_rate
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter
    with tf.variable_scope("inference"):
        inputs = image
        conv1_1 = Utils.conv(inputs, filters=64, l2_reg_scale=l2_reg, is_training=is_training)
        conv1_2 = Utils.conv(conv1_1, filters=64, l2_reg_scale=l2_reg, is_training=is_training)
        pool1 = Utils.pool(conv1_2)
        conv2_1 = Utils.conv(pool1, filters=128, l2_reg_scale=l2_reg, is_training=is_training)
        conv2_2 = Utils.conv(conv2_1, filters=128, l2_reg_scale=l2_reg, is_training=is_training)
        pool2 = Utils.pool(conv2_2)
        conv3_1 = Utils.conv(pool2, filters=256, l2_reg_scale=l2_reg, is_training=is_training)
        conv3_2 = Utils.conv(conv3_1, filters=256, l2_reg_scale=l2_reg, is_training=is_training)
        pool3 = Utils.pool(conv3_2)
        conv4_1 = Utils.conv(pool3, filters=512, l2_reg_scale=l2_reg, is_training=is_training)
        conv4_2 = Utils.conv(conv4_1, filters=512, l2_reg_scale=l2_reg, is_training=is_training)
        pool4 = Utils.pool(conv4_2)
        conv5_1 = Utils.conv(pool4, filters=1024, l2_reg_scale=l2_reg, is_training=is_training)
        conv5_2 = Utils.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg, is_training=is_training)
        concated1 = tf.concat([Utils.conv_transpose(conv5_2, filters=512, l2_reg_scale=l2_reg, is_training=is_training), conv4_2], axis=3)
        conv_up1_1 = Utils.conv(concated1, filters=512, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up1_2 = Utils.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg, is_training=is_training)
        concated2 = tf.concat([Utils.conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg, is_training=is_training), conv3_2], axis=3)
        conv_up2_1 = Utils.conv(concated2, filters=256, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up2_2 = Utils.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg, is_training=is_training)
        concated3 = tf.concat([Utils.conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg, is_training=is_training), conv2_2], axis=3)
        conv_up3_1 = Utils.conv(concated3, filters=128, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up3_2 = Utils.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg, is_training=is_training)
        concated4 = tf.concat([Utils.conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg, is_training=is_training), conv1_2], axis=3)
        conv_up4_1 = Utils.conv(concated4, filters=64, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up4_2 = Utils.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg, is_training=is_training)
        logits = Utils.conv(conv_up4_2, filters=NUM_OF_CLASSES, kernel_size=[1, 1], activation=None)
        annotation_pred = tf.argmax(logits, dimension=3, name="prediction")
        outputs = tf.expand_dims(annotation_pred, dim=3)
        return outputs, logits, net, conv5_2

def run():
    image = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    annotation = tf.placeholder(tf.int32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), name="annotation")
    training = tf.placeholder(tf.bool, shape=None, name="is_training")

    image075 = tf.image.resize_images(image, [int(IMAGE_SIZE * 0.75), int(IMAGE_SIZE * 0.75)])
    image050 = tf.image.resize_images(image, [int(IMAGE_SIZE * 0.5), int(IMAGE_SIZE * 0.5)])
    image125 = tf.image.resize_images(image, [int(IMAGE_SIZE * 1.25), int(IMAGE_SIZE * 1.25)])

    with tf.variable_scope('', reuse=False):
        pred_annotation100, logits100, net100, att100 = u_net_inference(image, is_training=False)
    with tf.variable_scope('', reuse=True):
        pred_annotation075, logits075, net075, att075 = u_net_inference(image075, is_training=False)
    with tf.variable_scope('', reuse=True):
        pred_annotation050, logits050, net050, att050 = u_net_inference(image050, is_training=False)
    with tf.variable_scope('', reuse=True):
        pred_annotation125, logits125, net125, att125 = u_net_inference(image125, is_training=False)

    attn_input = [att100,
                  tf.image.resize_images(att075, tf.shape(att100)[1:3]),
                  tf.image.resize_images(att050, tf.shape(att100)[1:3])]
    attn_input_train = tf.concat(attn_input, axis=3)
    attn_output_train = resnet.attn_m(attn_input_train, False)
    scale_att_mask = tf.nn.softmax(attn_output_train, axis=3)

    score_att_x = tf.multiply(logits100, tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 0], axis=3), tf.shape(logits100)[1:3]))
    score_att_x_075 = tf.multiply(tf.image.resize_images(logits075, tf.shape(logits100)[1:3]), tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 1], axis=3), tf.shape(logits100)[1:3]))
    score_att_x_050 = tf.multiply(tf.image.resize_images(logits050, tf.shape(logits100)[1:3]), tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 2], axis=3), tf.shape(logits100)[1:3]))
    score_att_x_125 = tf.multiply(tf.image.resize_images(logits125, tf.shape(logits100)[1:3]), tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 2], axis=3), tf.shape(logits100)[1:3]))
    score_final_train = score_att_x + score_att_x_075 + score_att_x_050 + score_att_x_125
    final_annotation_pred = tf.expand_dims(tf.argmax(score_final_train, dimension=3, name="final_prediction"), dim=3)

    test_records = fashion_parsing.read_datasetpred(data_dir)
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    test_dataset_reader = DataSetReader.BatchDatset(test_records, image_options)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("UNet model restored")
    else:
        print("ERROR: No UNet checkpoint found in", logs_dir)
        sess.close()
        return

    fd.mode_test(sess, batch_size, net100, test_dataset_reader,
                final_annotation_pred, score_final_train,
                image, annotation, training, None, None, None, saver)

    # Merge patches with overlap voting
    with open(%(image_list)r) as f:
        all_lines = [l.rstrip() for l in f if l.strip()]

    for line2 in all_lines:
        img = cv2.imread(line2, 0)
        img00 = cv2.imread(line2)
        (h, w, c) = img00.shape
        filename = data_dir + "/" + os.path.basename(os.path.splitext(line2)[0])

        annoimg_count_1 = np.zeros([h, w], dtype=np.float32)
        annoimg_count_total = np.zeros([h, w], dtype=np.float32)

        patch_size = 384
        stride = 344
        border = 10
        k = 0
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if (y_end - y) < 2 * border or (x_end - x) < 2 * border:
                    k += 1; continue
                patch_path = filename + "_" + str(k) + ".png"
                tmpimg = cv2.imread(patch_path, 0)
                if tmpimg is None:
                    k += 1; continue
                _, tmpimg = cv2.threshold(tmpimg, 0, 255, cv2.THRESH_BINARY)
                src_y_start = border
                src_y_end = min(patch_size - border, tmpimg.shape[0] - border)
                src_x_start = border
                src_x_end = min(patch_size - border, tmpimg.shape[1] - border)
                dest_y_start = y + border
                dest_y_end = y_end - border
                dest_x_start = x + border
                dest_x_end = x_end - border
                if (dest_y_end <= dest_y_start) or (dest_x_end <= dest_x_start):
                    k += 1; continue
                src_patch = tmpimg[src_y_start:src_y_end, src_x_start:src_x_end]
                src_patch = cv2.resize(src_patch, (dest_x_end - dest_x_start, dest_y_end - dest_y_start))
                try:
                    annoimg_count_1[dest_y_start:dest_y_end, dest_x_start:dest_x_end] += (src_patch / 255)
                    annoimg_count_total[dest_y_start:dest_y_end, dest_x_start:dest_x_end] += 1
                except Exception as e:
                    pass
                k += 1

        annoimg_ratio = np.zeros([h, w], dtype=np.float32)
        np.divide(annoimg_count_1, annoimg_count_total, out=annoimg_ratio, where=annoimg_count_total != 0)
        final_prediction = np.zeros([h, w], dtype=np.uint8)
        final_prediction[annoimg_ratio > 0.5] = 255

        if np.mean(img) > 240 and np.std(img) < 0.5:
            cv2.imwrite(filename + '.png', np.zeros([h, w], dtype=np.uint8))
        else:
            cv2.imwrite(filename + '.png', final_prediction)

    sess.close()
    print("UNet segmentation complete")

# ---- Main entry ----
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)

with open(%(image_list)r) as f:
    all_lines = [l.rstrip() for l in f if l.strip()]

for line in all_lines:
    img = cv2.imread(line)
    filename = data_dir + "/" + os.path.basename(os.path.splitext(line)[0])
    (h, w, c) = img.shape
    k = 0
    for i in range(int((h - 384) / 344) + 1):
        for j in range(int((w - 384) / 344) + 1):
            tmpimg = img[i * 344:(i * 344 + 384), j * 344:(j * 344 + 384)]
            cv2.imwrite(filename + "_" + str(k) + ".jpg", tmpimg)
            k += 1
        tmpimg = img[i * 344:(i * 344 + 384), (w - 384):w]
        cv2.imwrite(filename + "_" + str(k) + ".jpg", tmpimg)
        k += 1
    for j in range(int((w - 384) / 344) + 1):
        tmpimg = img[(h - 384):h, j * 344:(j * 344 + 384)]
        cv2.imwrite(filename + "_" + str(k) + ".jpg", tmpimg)
        k += 1
    tmpimg = img[(h - 384):h, (w - 384):w]
    cv2.imwrite(filename + "_" + str(k) + ".jpg", tmpimg)

run()
''' % {
        'modules_dir': modules_dir,
        'gpu_id': gpu_id,
        'logs_dir': logs_dir,
        'pred_dir': pred_dir,
        'image_list': image_list_path,
        'tf_python_cmd_str': tf_python_cmd_str,
    }

    script_path = os.path.join(work_dir, 'run_unet_seg.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path


def run_unet_seg(image_paths, work_dir, gpu_id=0):
    """
    Run UNet+Attention nuclear segmentation on a list of images via subprocess
    (with correct glibc loader).

    Args:
        image_paths: list of full paths to input images (.png/.jpg)
        work_dir: temporary working directory
        gpu_id: GPU device id (int)

    Returns:
        dict mapping basename -> mask (numpy HxW, uint8, 0 or 255)
    """
    import subprocess

    pred_dir = os.path.join(work_dir, 'unet_pred')
    os.makedirs(pred_dir, exist_ok=True)

    # Generate the standalone UNet script
    script_path = _generate_unet_script(image_paths, work_dir, gpu_id)

    # Use the glibc loader from bashrc alias to launch subprocess
    tf_python_cmd = _get_tf_python_cmd()

    print("  Running UNet segmentation (TensorFlow subprocess)...")
    result = subprocess.run(
        tf_python_cmd + [script_path],
        capture_output=True,
        text=True,
        cwd=work_dir,
        timeout=3600
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
        print("  UNet stderr (last 3000 chars):", stderr_tail)
        raise RuntimeError("UNet segmentation failed with return code %d" % result.returncode)
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print("  ", line)

    # Read results
    results = {}
    for p in image_paths:
        basename = os.path.splitext(os.path.basename(p))[0]
        mask_path = os.path.join(pred_dir, basename + '.png')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            if mask is not None:
                results[basename] = mask

    return results
