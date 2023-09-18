
import os
import logging
import torch
import struct
import numpy as np
import tensorrt as trt
from structures.register import Registry


TRT_BACKBONE_REGISTRY = Registry("TRT_BACKBONE")
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def load_weights(file):
    assert os.path.exists(file), 'Unable to load weight file.'
    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)
    LOGGER.info(f"success Loading .wts weights: {file}")
    return weight_map


def conv_engine(weight_map, cfg):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = cfg["MAX_BATCH_SIZE"]
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # (1G)
    # create network layers
    network = builder.create_network()
    network = TRT_BACKBONE_REGISTRY.get(
        cfg["BACKBONE"])(weight_map, network, cfg)
    # Build Engine
    engine = builder.build_engine(network.forward(), config)

    with open(cfg["ENGINE_PATH"], "wb") as f:
        f.write(engine.serialize())

    del network
    del weight_map
    del engine
    del builder
    del config


def conv_wts(cfg):
    model = torch.load(cfg["PT_PATH"], map_location=cfg["DEVICE"])
    LOGGER.info(f"success load .pt model:{cfg['PT_PATH']}")
    f = open(cfg["WTS_PATH"], 'w')

    if not isinstance(model, dict):
        f.write("{}\n".format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().detach().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
    else:
        f.write("{}\n".format(len(model.keys())))
        for k, v in model.items():
            vr = v.reshape(-1).cpu().detach().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
    LOGGER.info(f"success convert .pt model to .wts model:{cfg['WTS_PATH']}")
