#!/bin/bash

dora run solver=VidMuse/VidMuse_example \
        continue_from=//pretrained/facebook/musicgen-small \
        model/lm/model_scale=small