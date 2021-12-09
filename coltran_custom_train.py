# %% INIT
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import datetime

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


### INIT [USER INPUT]:
MAX_TRAIN = 100
OUTPUT_DIR = 'coltran/log-{}'.format(TAG)

# %% Definitions:
from enum import Enum

class COLORTRAN_STEPS(Enum):
    INIT               = 0
    COLORIZER          = 1
    COPLOR_UPSAMPLER   = 2
    SPATIAL_UPSAMPLER  = 3

class COLORTRAN_FEATURE(Enum):
    BASELINE    = 0 # baseline axial trnasformer without conditioning
    NO_cLN      = 1
    NO_cMLP     = 2
    NO_cAtt     = 3
    CONDITIONAL = 4

CONFIG = {
    COLORTRAN_STEPS.COLORIZER: {
        "image_directory": 'coltran/result-train/imagenet/color',
        "model_config": CONFIG_COLTRAN_CORE,
        "batch_size": 25, #20
        # "pre-built_log_dir": 'coltran/coltran/colorizer',
        # "output_path": 'coltran/result{}/stage_1'.format(TAG),
        "output_log": '{}/stage_1'.format(OUTPUT_DIR),
        "steps_per_summaries": MAX_TRAIN/100,
        "max_train_steps": MAX_TRAIN,
    },
    # COLORTRAN_STEPS.COPLOR_UPSAMPLER: {
    #     "image_directory": 'coltran/result-train/imagenet/color',
    #     "model_config": CONFIG_COLOR_UPSAMPLER,
    #     "batch_size": 5,
    #     # "pre-built_log_dir": 'coltran/coltran/color_upsampler',
    #     # "output_path": 'coltran/result{}/stage_2'.format(TAG),
    #     "output_log": '{}/stage_2'.format(OUTPUT_DIR),
    #     "steps_per_summaries": MAX_TRAIN/10,
    #     "max_train_steps": MAX_TRAIN,
    # },
    # COLORTRAN_STEPS.SPATIAL_UPSAMPLER: {
    #     "image_directory": 'coltran/result-train/imagenet/color',
    #     "model_config": CONFIG_SPATIAL_UPSAMPLER,
    #     "batch_size": 5,
    #     # "pre-built_log_dir": 'coltran/coltran/spatial_upsampler',
    #     # "output_path": 'coltran/result{}/stage_3'.format(TAG),
    #     "output_log": '{}/stage_3'.format(OUTPUT_DIR),
    #     "steps_per_summaries": MAX_TRAIN/10,
    #     "max_train_steps": MAX_TRAIN,
    # },
}

# define logger:
def _print(content):
    print("[COLTRAIN_TRAIN] ", content)
    with open(os.path.join(OUTPUT_DIR,"log.txt"), "a") as log_file:
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
## Train.
###############################################################################
def train(model_step, logdir, config):
    steps_per_write = CONFIG[model_step]["steps_per_summaries"]
    
    _print("================================ TRAINING MODEL [{}] ================================".format(model_step))
    
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
    start_time = time.time()
    total_time = time.time()

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

    t_steps = optimizer.iterations.numpy()
    N_steps = min(config.get('max_train_steps', 1000), CONFIG[model_step]["max_train_steps"])
    # TRAINING: 
    while t_steps < N_steps:
        num_train_steps = optimizer.iterations 
        t_steps = optimizer.iterations.numpy()

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

        if time.time() - start_time > config.save_checkpoint_secs:
            checkpoint_name = checkpoint.save()
            _print('Saved checkpoint to {}'.format(checkpoint_name))
            start_time = time.time()

    total_time = time.time() - total_time
    
    _print("================================ MODEL END @ [{}] ================================".format(model_step))
    return train_loss, total_time


# %%
def main():
    # MAIN:
    jx_lib.create_all_folders(OUTPUT_DIR)
    tf.keras.backend.clear_session()
    _print(RUN_STEPS)

    loss = {}
    time = {}
    ### step 1:
    if COLORTRAN_STEPS.COLORIZER in RUN_STEPS:
        
        config = CONFIG[COLORTRAN_STEPS.COLORIZER]["model_config"].get_config()
        logdir = CONFIG[COLORTRAN_STEPS.COLORIZER]["output_log"]

        if "ablation" in TAG:
            def get_modified_dir_and_config(feature):
                config = CONFIG[COLORTRAN_STEPS.COLORIZER]["model_config"].get_config()
                logdir = CONFIG[COLORTRAN_STEPS.COLORIZER]["output_log"]
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
                    logdir = os.path.join(logdir, 'no_cAtt')
                else:
                    # do nothing
                    pass
                return logdir, config

            for feature in COLORTRAN_FEATURE:
                # modify flags
                logdir, config = get_modified_dir_and_config(feature=feature)
                ic(logdir)
                # train:
                loss[feature], time[feature] = \
                    train(
                        model_step = COLORTRAN_STEPS.COLORIZER,
                        logdir = logdir, config = config
                    )
                tf.keras.backend.clear_session()
        else:
            loss[COLORTRAN_STEPS.COLORIZER], time[COLORTRAN_STEPS.COLORIZER] = \
                train(
                    model_step = COLORTRAN_STEPS.COLORIZER,
                    logdir = logdir, config = config
                )
            tf.keras.backend.clear_session()

    # ### step 2:
    # if COLORTRAN_STEPS.COPLOR_UPSAMPLER in RUN_STEPS:
    #     loss[COLORTRAN_STEPS.COPLOR_UPSAMPLER], time[COLORTRAN_STEPS.COPLOR_UPSAMPLER] = \
    #         train(model_step = COLORTRAN_STEPS.COPLOR_UPSAMPLER)
    #     tf.keras.backend.clear_session()

    # ### step 3:
    # if COLORTRAN_STEPS.SPATIAL_UPSAMPLER in RUN_STEPS:
    #     loss[COLORTRAN_STEPS.SPATIAL_UPSAMPLER], time[COLORTRAN_STEPS.SPATIAL_UPSAMPLER] = \
    #         train(model_step = COLORTRAN_STEPS.SPATIAL_UPSAMPLER)
    #     tf.keras.backend.clear_session()

    # log:
    _print("[Ellapsed Time]: {}".format(time))
    jx_lib.output_plot(
        data_dict = loss,
        Ylabel    = "Training Loss",
        Xlabel    = "Training Steps",
        OUT_DIR   = "output",
        tag       = "training_loss_({})".format(TAG),
    )


# %%
if __name__ == "__main__":
    main()