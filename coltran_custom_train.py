# %% INIT
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import datetime
import json

from coltran import datasets
from coltran.models import colorizer
from coltran.models import upsampler
from coltran.utils import base_utils
from coltran.utils import datasets_utils
from coltran.utils import train_utils
import collections
import functools

import tensorflow as tf
import tensorflow_datasets as tfds

from coltran_configs import CONFIG_COLTRAN_CORE, CONFIG_COLOR_UPSAMPLER, CONFIG_SPATIAL_UPSAMPLER

import jx_lib
from icecream import ic

# %% USER PARAMS:
TAG = ""
if len(sys.argv) > 1:
    TAG = sys.argv[1]
    # TAG = "-potato"
    # TAG = "-batch1"
else:
    raise ValueError("Require One Argument")

if "eval" in TAG:
    MODE = "validation"
elif "train" in TAG:
    MODE = "training"
else:
    raise ValueError("Invalid TAG !!!")

MAX_TRAIN = 100
STEP_SIZE = 10
SAVING_RATE = 1
if "-10K" in TAG:
    MAX_TRAIN = 10000
    STEP_SIZE = 100
if "-20K" in TAG:
    MAX_TRAIN = 20000
    STEP_SIZE = 100
if "-1K" in TAG:
    MAX_TRAIN = 1000
    STEP_SIZE = 10
if "-500" in TAG:
    MAX_TRAIN = 500
    STEP_SIZE = 10

TAG = TAG.replace('-eval', '')

OUTPUT_TRN_DIR = 'coltran/log-{}'.format(TAG)
OUTPUT_VAL_DIR = 'coltran/log-{}-eval'.format(TAG)

ic(MAX_TRAIN)
ic(OUTPUT_TRN_DIR)
ic(OUTPUT_VAL_DIR)

# %% Definitions:
from enum import Enum

class COLORTRAN_STEPS(Enum):
    INIT               = 0
    COLORIZER          = 1
    COPLOR_UPSAMPLER   = 2
    SPATIAL_UPSAMPLER  = 3

class COLORTRAN_FEATURE(Enum):
    BASELINE    = "Baseline" # baseline axial trnasformer without conditioning
    NO_cLN      = "No cLN"
    NO_cMLP     = "No cMLP"
    NO_cAtt     = "No cAtt"
    CONDITIONAL = "ColTran"

CONFIG = {
    COLORTRAN_STEPS.COLORIZER: {
        "image_directory": 'coltran/result-train/imagenet/color',
        "model_config": CONFIG_COLTRAN_CORE,
        "batch_size": 20, #20
        # "pre-built_log_dir": 'coltran/coltran/colorizer',
        # "output_path": 'coltran/result{}/stage_1'.format(TAG),
        "output_training_log": '{}/stage_1'.format(OUTPUT_TRN_DIR),
        "steps_per_summaries": STEP_SIZE, # => NOTE: we want more than 100 poitns for best result on graphing
        "max_train_steps": MAX_TRAIN,
        "validation_image_directory": 'coltran/result-val-1k/imagenet/color',
    },
    COLORTRAN_STEPS.COPLOR_UPSAMPLER: {
        "image_directory": 'coltran/result-train/imagenet/color',
        "model_config": CONFIG_COLOR_UPSAMPLER,
        "batch_size": 5,
        # "pre-built_log_dir": 'coltran/coltran/color_upsampler',
        # "output_path": 'coltran/result{}/stage_2'.format(TAG),
        "output_training_log": '{}/stage_2'.format(OUTPUT_TRN_DIR),
        "steps_per_summaries": STEP_SIZE,
        "max_train_steps": MAX_TRAIN,
    },
    COLORTRAN_STEPS.SPATIAL_UPSAMPLER: {
        "image_directory": 'coltran/result-train/imagenet/color',
        "model_config": CONFIG_SPATIAL_UPSAMPLER,
        "batch_size": 5,
        # "pre-built_log_dir": 'coltran/coltran/spatial_upsampler',
        # "output_path": 'coltran/result{}/stage_3'.format(TAG),
        "output_training_log": '{}/stage_3'.format(OUTPUT_TRN_DIR),
        "steps_per_summaries": STEP_SIZE,
        "max_train_steps": MAX_TRAIN,
    },
}

# define logger:
def _print(content):
    print("[COLTRAIN_TRAIN] ", content)
    with open(os.path.join(OUTPUT_TRN_DIR,"log[{}].txt".format(TAG)), "a") as log_file:
        log_file.write("\n")
        log_file.write("[{}]: {}".format(datetime.datetime.now(), content))

RUN_STEPS = []
if len(sys.argv) > 2:
    if "colorizer" in sys.argv:
        RUN_STEPS.append(COLORTRAN_STEPS.COLORIZER)
    if "color_upsampler" in sys.argv:
        RUN_STEPS.append(COLORTRAN_STEPS.COPLOR_UPSAMPLER)
    if "spatial_upsampler" in sys.argv:
        RUN_STEPS.append(COLORTRAN_STEPS.SPATIAL_UPSAMPLER)
else:
    RUN_STEPS = [
        COLORTRAN_STEPS.INIT,
        COLORTRAN_STEPS.COLORIZER,
        COLORTRAN_STEPS.COPLOR_UPSAMPLER,
        COLORTRAN_STEPS.SPATIAL_UPSAMPLER,
    ]

# %%



def loss_on_batch(inputs, model, config, training=False):
    """Loss on a batch of inputs."""
    logits, aux_output = model.get_logits(
        inputs_dict=inputs, train_config=config, training=training)
    loss, aux_loss_dict = model.loss(
        targets=inputs, logits=logits, train_config=config, training=training,
        aux_output=aux_output)
    loss_factor = config.get('loss_factor', 1.0)

    loss_dict = collections.OrderedDict()
    loss_dict['loss'] = loss
    total_loss = loss_factor * loss

    for aux_key, aux_loss in aux_loss_dict.items():
        aux_loss_factor = config.get(f'{aux_key}_loss_factor', 1.0)
        loss_dict[aux_key] = aux_loss
        total_loss += aux_loss_factor * aux_loss
    loss_dict['total_loss'] = total_loss

    extra_info = collections.OrderedDict([
        ('scalar', loss_dict),
    ])
    return total_loss, extra_info


def train_step(config,
            model,
            optimizer,
            metrics,
            ema=None,
            strategy=None):
    """Training StepFn."""

    def step_fn(inputs):
        """Per-Replica StepFn."""
        with tf.GradientTape() as tape:
            loss, extra = loss_on_batch(inputs, model, config, training=True)
            scaled_loss = loss
            if strategy:
                scaled_loss /= float(strategy.num_replicas_in_sync)

        grads = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        for metric_key, metric in metrics.items():
            metric.update_state(extra['scalar'][metric_key])

        if ema is not None:
            ema.apply(model.trainable_variables)
        return loss

    return train_utils.step_with_strategy(step_fn, strategy)


def build_model(
        model_step: COLORTRAN_STEPS,
    ):
    # same as the one in custom_run
    """Builds model."""
    config = CONFIG[model_step]["model_config"].get_config()
    _print("[{}] > model config: {}".format(model_step, config.model.name))
    optimizer = train_utils.build_optimizer(config)

    zero_64 = tf.zeros((1, 64, 64, 3), dtype=tf.int32)
    zero_64_slice = tf.zeros((1, 64, 64, 1), dtype=tf.int32)
    zero = tf.zeros((1, 256, 256, 3), dtype=tf.int32)
    zero_slice = tf.zeros((1, 256, 256, 1), dtype=tf.int32)

    if model_step is COLORTRAN_STEPS.COLORIZER:
        model = colorizer.ColTranCore(config.model)
        model(zero_64, training=False)
    
    elif model_step is COLORTRAN_STEPS.COPLOR_UPSAMPLER:
        model = upsampler.ColorUpsampler(config.model)
        model(inputs=zero_64, inputs_slice=zero_64_slice, training=False)
    
    elif model_step is COLORTRAN_STEPS.SPATIAL_UPSAMPLER:
        model = upsampler.SpatialUpsampler(config.model)
        model(inputs=zero, inputs_slice=zero_slice, training=False)

    ema_vars = model.trainable_variables
    ema = train_utils.build_ema(config, ema_vars)
    return model, optimizer, ema

def restore_checkpoint(model, ema, strategy, latest_ckpt=None, optimizer=None):
    if optimizer is None:
        ckpt_func = functools.partial(
            train_utils.create_checkpoint, models=model, ema=ema)
    else:
        ckpt_func = functools.partial(
            train_utils.create_checkpoint, models=model, ema=ema,
            optimizer=optimizer)

    checkpoint = train_utils.with_strategy(ckpt_func, strategy)
    if latest_ckpt:
        _print('Restoring from pretrained directory: {}'.format(latest_ckpt))
        train_utils.with_strategy(lambda: checkpoint.restore(latest_ckpt), strategy)
    return checkpoint

###############################################################################
## Evaluating.
###############################################################################


def evaluate(model_step, logdir, config, feature, subset="valid"):
    total_time = time.time()
    _print("================================ Evaluating MODEL [{}:{}] ================================".format(model_step, feature))
    
    steps_per_write = CONFIG[model_step]["steps_per_summaries"]
    """Executes the evaluation loop."""

    strategy, batch_size = train_utils.setup_strategy(config, "local", 1, "eval_valid", 'GPU')

    def input_fn(input_context=None):
        read_config = None
        if input_context is not None:
            read_config = tfds.ReadConfig(input_context=input_context)
        dataset = datasets.get_dataset(
            name='custom',
            config=config,
            batch_size=config.batch_size,
            subset='train',
            read_config=read_config,
            data_dir=CONFIG[model_step]["validation_image_directory"])
        return dataset

    model, optimizer, ema = train_utils.with_strategy(lambda: build_model(model_step=model_step), strategy)

    metric_keys = ['loss', 'total_loss']
    # metric_keys += model.metric_keys
    metrics = {}
    for metric_key in metric_keys:
        func = functools.partial(tf.keras.metrics.Mean, metric_key)
        curr_metric = train_utils.with_strategy(func, strategy)
        metrics[metric_key] = curr_metric

    checkpoints = train_utils.with_strategy(
        lambda: train_utils.create_checkpoint(model, optimizer, ema),
        strategy)
    dataset = train_utils.dataset_with_strategy(input_fn, strategy)
    # ic(dataset)
    def step_fn(batch):
        _, extra = loss_on_batch(batch, model, config, training=False)

        for metric_key in metric_keys:
            curr_metric = metrics[metric_key]
            curr_scalar = extra['scalar'][metric_key]
            curr_metric.update_state(curr_scalar)

    num_examples = config.eval_num_examples
    eval_step = train_utils.step_with_strategy(step_fn, strategy)
    ckpt_path = None
    wait_max = 100
    is_ema = True if ema else False

    eval_summary_dir = os.path.join(
        logdir, 'eval_{}_summaries_pyk_{}'.format(subset, is_ema))
    writer = tf.summary.create_file_writer(eval_summary_dir)

    eval_loss = {
        "x": [],
        "y": [],
    }

    N_models = int(MAX_TRAIN/STEP_SIZE)
    for count in range(N_models):
        while_time = time.time()
        ckpt_path = os.path.join(logdir, "model-{}".format(count + 3))
        _print("=== Evaluating: {}".format(ckpt_path))
        if ckpt_path is None:
            _print('Timed out waiting for checkpoint.')
            break

        train_utils.with_strategy(lambda: train_utils.restore(model, checkpoints, logdir, ema),strategy)
        data_iterator = iter(dataset)
        num_steps = num_examples // batch_size

        for metric_key, metric in metrics.items():
            metric.reset_states()

        _print('Starting evaluation for {} steps ...'.format(num_steps))
        done = False
        for i in range(0, num_steps, steps_per_write):
            start_run = time.time()
            for k in range(min(num_steps - i, steps_per_write)):
                try:
                    if k % 10 == 0:
                        _print('Step: {}'.format(i + k + 1))
                    eval_step(data_iterator)
                except (StopIteration, tf.errors.OutOfRangeError):
                    done = True
                break
            if done:
                break
            steps_per_sec =  (time.time() - start_run)/steps_per_write
            bits_per_dim = metrics['loss'].result()
            _print('[{:5d}/{}] Loss: {:.3f} bits/dim, Speed: {:.3f} steps/second, Ellapsed: {:.3f} seconds'.format(
                        i, num_steps, bits_per_dim, steps_per_sec, time.time() - start_run))

        with writer.as_default():
            for metric_key, metric in metrics.items():
                curr_scalar = metric.result().numpy()
                t_steps = optimizer.iterations.numpy()
                tf.summary.scalar(metric_key, curr_scalar, step=optimizer.iterations)
                
                if metric_key == 'total_loss':
                    _print('=====> [Writer] Total Loss: {:.3f} bits/dim, Step: {}, Total Ellapsed: {:.3f} seconds'.format(
                        curr_scalar, t_steps, time.time() - while_time
                    ))
                if metric_key == 'loss':
                    _print('=====> [Writer] Loss: {:.3f} bits/dim, Step: {}, Total Ellapsed: {:.3f} seconds'.format(
                        curr_scalar, t_steps, time.time() - while_time
                    ))
                    eval_loss["x"].append(count * SAVING_RATE + 1)
                    eval_loss["y"].append(curr_scalar)
        
        count += 1
        
    _print("================================ MODEL END @ [{}:{}] ================================".format(model_step, feature))
    return eval_loss, (time.time() - total_time)


def evaluate_during_train(model_step, logdir, config, model, num_train_steps, subset="valid"):
    total_time = time.time()
    _print("================================ Evaluating MODEL [{}:{}] ================================".format(model_step, num_train_steps))
    
    steps_per_write = CONFIG[model_step]["steps_per_summaries"]
    """Executes the evaluation loop."""

    strategy, batch_size = train_utils.setup_strategy(config, "local", 1, "eval_valid", 'GPU')

    def input_fn(input_context=None):
        read_config = None
        if input_context is not None:
            read_config = tfds.ReadConfig(input_context=input_context)
        dataset = datasets.get_dataset(
            name='custom',
            config=config,
            batch_size=config.batch_size,
            subset='train',
            read_config=read_config,
            data_dir=CONFIG[model_step]["validation_image_directory"])
        return dataset


    metric_keys = ['loss', 'total_loss']
    # metric_keys += model.metric_keys
    metrics = {}
    for metric_key in metric_keys:
        func = functools.partial(tf.keras.metrics.Mean, metric_key)
        curr_metric = train_utils.with_strategy(func, strategy)
        metrics[metric_key] = curr_metric

    dataset = train_utils.dataset_with_strategy(input_fn, strategy)
    # ic(dataset)
    def step_fn(batch):
        _, extra = loss_on_batch(batch, model, config, training=False)

        for metric_key in metric_keys:
            curr_metric = metrics[metric_key]
            curr_scalar = extra['scalar'][metric_key]
            curr_metric.update_state(curr_scalar)

    num_examples = config.eval_num_examples
    eval_step = train_utils.step_with_strategy(step_fn, strategy)

    eval_summary_dir = os.path.join(
        logdir, 'eval_{}_summaries_pyk_{}'.format(subset, True))
    writer = tf.summary.create_file_writer(eval_summary_dir)

    eval_loss = {}
    data_iterator = iter(dataset)
    num_steps = num_examples // batch_size

    for metric_key, metric in metrics.items():
        metric.reset_states()

    _print('Starting evaluation for {} steps ...'.format(num_steps))
    done = False
    for i in range(0, num_steps, steps_per_write):
        start_run = time.time()
        for k in range(min(num_steps - i, steps_per_write)):
            try:
                if k % 10 == 0:
                    _print('Step: {}'.format(i + k + 1))
                eval_step(data_iterator)
            except (StopIteration, tf.errors.OutOfRangeError):
                done = True
            break
        if done:
            break
        steps_per_sec =  (time.time() - start_run)/steps_per_write
        bits_per_dim = metrics['loss'].result()
        _print('[{:5d}/{}] Validation Loss: {:.3f} bits/dim, Speed: {:.3f} steps/second, Ellapsed: {:.3f} seconds'.format(
                    i, num_steps, bits_per_dim, steps_per_sec, time.time() - start_run))

    with writer.as_default():
        for metric_key, metric in metrics.items():
            curr_scalar = metric.result().numpy()
            tf.summary.scalar(metric_key, curr_scalar, step=num_train_steps)
            eval_loss[metric_key] = curr_scalar
        
    _print("================================ MODEL VALIDATION END @ [{}:{}] ================================".format(model_step, num_train_steps))
    return eval_loss

###############################################################################
## Train.
###############################################################################
def train(model_step, logdir, config, feature="None", if_save_checkpoint = True, if_evaluate = False):
    steps_per_write = CONFIG[model_step]["steps_per_summaries"]
    total_time = time.time()
    _print("================================ TRAINING MODEL [{}:{}] ================================".format(model_step, feature))
    
    # save config
    train_utils.write_config(config, logdir)

    strategy, batch_size = train_utils.setup_strategy(config, "local", 1, "train", 'GPU')

    def input_fn(input_context=None):
        read_config = None
        if input_context is not None:
            read_config = tfds.ReadConfig(input_context=input_context)
        dataset = datasets.get_dataset(
            name='custom',
            config=config,
            batch_size=config.batch_size,
            subset='train',
            read_config=read_config,
            data_dir=CONFIG[model_step]["image_directory"])
        return dataset

    # DATASET CREATION.
    _print('Building dataset.')
    train_dataset = train_utils.dataset_with_strategy(input_fn, strategy)
    data_iterator = iter(train_dataset)

    # MODEL BUILDING
    _print('Building model.')
    model, optimizer, ema = train_utils.with_strategy(lambda: build_model(model_step=model_step), strategy)
    model.summary(120, print_fn=_print)

    # METRIC CREATION.
    metrics = {}
    metric_keys = ['loss', 'total_loss']
    metric_keys += model.metric_keys
    for metric_key in metric_keys:
        func = functools.partial(tf.keras.metrics.Mean, metric_key)
        curr_metric = train_utils.with_strategy(func, strategy)
        metrics[metric_key] = curr_metric

    # # CHECKPOINTING LOGIC.
    # if FLAGS.pretrain_dir is not None:
    #     pretrain_ckpt = tf.train.latest_checkpoint(FLAGS.pretrain_dir)
    #     assert pretrain_ckpt
    # # Load the entire model without the optimizer from the checkpoints.
    # restore_checkpoint(model, ema, strategy, pretrain_ckpt, optimizer=None)
    # # New tf.train.Checkpoint instance with a reset optimizer.
    # checkpoint = restore_checkpoint(
    #     model, ema, strategy, latest_ckpt=None, optimizer=optimizer)
    # else:
    
    # # INIT CHECKPOINT:
    latest_ckpt = tf.train.latest_checkpoint(logdir)
    checkpoint = restore_checkpoint(
        model, ema, strategy, latest_ckpt, optimizer=optimizer)

    checkpoint = tf.train.CheckpointManager(
        checkpoint, directory=logdir, checkpoint_name='model', max_to_keep=10)
    if optimizer.iterations.numpy() == 0:
        checkpoint_name = checkpoint.save()
        _print('Saved checkpoint to '.format(checkpoint_name))

    train_summary_dir = os.path.join(logdir, 'train_summaries')
    writer = tf.summary.create_file_writer(train_summary_dir)

    _print('Start Training.')

    # This hack of wrapping up multiple train steps with a tf.function call
    # speeds up training significantly.
    # See: https://www.tensorflow.org/guide/tpu#improving_performance_by_multiple_steps_within_tffunction # pylint: disable=line-too-long
    @tf.function
    def train_multiple_steps(iterator, steps_per_epoch):
        train_step_f = train_step(config, model, optimizer, metrics, ema,
                                    strategy)
        for _ in range(steps_per_epoch):
            train_step_f(iterator)

    train_loss = {
        "x": [],
        "y": [],
    }
    valid_loss = {
        "x": [],
        "y": [],
    }

    t_steps = optimizer.iterations.numpy()
    N_steps = min(config.get('max_train_steps', 1000), CONFIG[model_step]["max_train_steps"])
    # TRAINING: 
    while t_steps < N_steps:
        start_time = time.time()
        num_train_steps = optimizer.iterations 
        t_steps = num_train_steps.numpy()

        for metric_key in metric_keys:
            metrics[metric_key].reset_states()

        start_run = time.time()

        train_multiple_steps(data_iterator, tf.convert_to_tensor(steps_per_write))

        steps_per_sec = steps_per_write / (time.time() - start_run)
        with writer.as_default():
            for metric_key, metric in metrics.items():
                metric_np = metric.result().numpy()
                # _print("[{:5d}] {:8s}:{}".format(t_steps, metric_key, metric_np))
                tf.summary.scalar(metric_key, metric_np, step=num_train_steps)

                if metric_key == 'total_loss':
                    _print('[{:5d}/{}] Loss: {:.3f} bits/dim, Speed: {:.3f} steps/second, Ellapsed: {:.3f} seconds'.format(
                        t_steps, N_steps, metric_np, steps_per_sec, time.time() - start_time))
                
                if metric_key == 'loss':
                    train_loss["x"].append(t_steps)
                    train_loss["y"].append(metric_np)

        if int(t_steps % SAVING_RATE) == 0:
            if if_save_checkpoint:
                checkpoint_name = checkpoint.save()
                _print('Saved checkpoint to {}'.format(checkpoint_name))
            if if_evaluate:
                eval_loss = evaluate_during_train(model_step, logdir, config, model, num_train_steps, subset="valid")
                ic(eval_loss)
                valid_loss["x"].append(t_steps)
                valid_loss["y"].append(eval_loss['loss'])


    total_time = time.time() - total_time
    
    _print("================================ MODEL TRAINING END @ [{}:{}] ================================".format(model_step,feature))
    return train_loss, valid_loss, total_time


# %%
def main():
    # MAIN:
    jx_lib.create_all_folders(OUTPUT_TRN_DIR)
    tf.keras.backend.clear_session()
    _print(RUN_STEPS)

    loss = {}
    valid_loss = {}
    time = {}
    ### step 1:
    if COLORTRAN_STEPS.COLORIZER in RUN_STEPS:
        
        config = CONFIG[COLORTRAN_STEPS.COLORIZER]["model_config"].get_config()
        logdir = CONFIG[COLORTRAN_STEPS.COLORIZER]["output_training_log"]

        if "ablation" in TAG:
            def get_modified_dir_and_config(feature):
                config = CONFIG[COLORTRAN_STEPS.COLORIZER]["model_config"].get_config()
                logdir = CONFIG[COLORTRAN_STEPS.COLORIZER]["output_training_log"]
                ic(feature)
                if feature is COLORTRAN_FEATURE.BASELINE:
                    config.model.decoder.mlp            = ''
                    config.model.decoder.cond_ln        = False
                    config.model.decoder.cond_att_v     = False
                    config.model.decoder.cond_att_q     = False
                    config.model.decoder.cond_att_k     = False
                    config.model.decoder.cond_att_scale = False
                    logdir = os.path.join(logdir, 'baseline')
                elif feature is COLORTRAN_FEATURE.NO_cLN:
                    config.model.decoder.cond_ln = False
                    logdir = os.path.join(logdir, 'no_cLN')
                elif feature is COLORTRAN_FEATURE.NO_cMLP:
                    config.model.decoder.mlp = ''
                    logdir = os.path.join(logdir, 'no_cMLP')
                elif feature is COLORTRAN_FEATURE.NO_cAtt:
                    config.model.decoder.cond_att_v = False
                    config.model.decoder.cond_att_q = False
                    config.model.decoder.cond_att_k = False
                    config.model.decoder.cond_att_scale = False
                    logdir = os.path.join(logdir, 'no_cAtt')
                else:
                    # do nothing
                    pass
                return logdir, config

                
            for feature in COLORTRAN_FEATURE:
                # modify flags
                logdir, config = get_modified_dir_and_config(feature=feature)
                ic(logdir)
                if MODE == "training":
                    # train:
                    loss[feature.value], valid_loss[feature.value], time[feature.value] = \
                        train(
                            model_step = COLORTRAN_STEPS.COLORIZER,
                            logdir = logdir, config = config, feature = feature,
                            if_evaluate = ("validate" in TAG)
                        )
                    tf.keras.backend.clear_session()    
                elif MODE == "validation":
                    valid_loss[feature.value], time[feature.value] = \
                        evaluate(
                            model_step = COLORTRAN_STEPS.COLORIZER,
                            logdir = logdir, config = config, feature = feature
                        )
        else:
            loss[COLORTRAN_STEPS.COLORIZER.value], _, time[COLORTRAN_STEPS.COLORIZER.value] = \
                train(
                    model_step = COLORTRAN_STEPS.COLORIZER,
                    logdir = logdir, config = config
                )
            tf.keras.backend.clear_session()

    if "ablation" not in TAG:
        ### step 2:
        if COLORTRAN_STEPS.COPLOR_UPSAMPLER in RUN_STEPS:
            config = CONFIG[COLORTRAN_STEPS.COPLOR_UPSAMPLER]["model_config"].get_config()
            logdir = CONFIG[COLORTRAN_STEPS.COPLOR_UPSAMPLER]["output_training_log"]
            loss[COLORTRAN_STEPS.COPLOR_UPSAMPLER], _, time[COLORTRAN_STEPS.COPLOR_UPSAMPLER] = \
                train(model_step = COLORTRAN_STEPS.COPLOR_UPSAMPLER, logdir = logdir, config = config)
            tf.keras.backend.clear_session()

        ### step 3:
        if COLORTRAN_STEPS.SPATIAL_UPSAMPLER in RUN_STEPS:
            config = CONFIG[COLORTRAN_STEPS.SPATIAL_UPSAMPLER]["model_config"].get_config()
            logdir = CONFIG[COLORTRAN_STEPS.SPATIAL_UPSAMPLER]["output_training_log"]
            loss[COLORTRAN_STEPS.SPATIAL_UPSAMPLER], _, time[COLORTRAN_STEPS.SPATIAL_UPSAMPLER] = \
                train(model_step = COLORTRAN_STEPS.SPATIAL_UPSAMPLER, logdir = logdir, config = config)
            tf.keras.backend.clear_session()

    # log:
    _print("[Ellapsed Time]: {}".format(time))
    def plot_loss(loss, MODE):
        # plot 4 losses diagarms
        for item in [None, "ewm", "savgol_filter"]:
            try:
                jx_lib.output_plot(
                    data_dict = loss,
                    Ylabel    = "{} Loss".format(MODE),
                    Xlabel    = "{} Steps".format(MODE),
                    OUT_DIR   = "output",
                    tag       = "{}_loss_({})".format(MODE, TAG),
                    smooth    = item
                )
            except:
                _print("[ERROR] Unable to Plot {}".format(item))
    if len(valid_loss) > 0:
        plot_loss(loss=valid_loss, MODE="validation")
    if len(loss) > 0:
        plot_loss(loss=loss, MODE="training")

# %%
if __name__ == "__main__":
    main()