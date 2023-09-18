import yaml
from backbone.build_trt import conv_wts, conv_engine, load_weights

if __name__ == "__main__":
    cfg = yaml.safe_load(open('trt.yaml'))
    conv_wts(cfg)
    weight_map = load_weights(cfg["WTS_PATH"])
    print(weight_map.keys())
    conv_engine(weight_map, cfg)
