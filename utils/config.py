import os, yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            if isinstance(value, dict):
                value = AttrDict(value)
            elif isinstance(value, list):
                value = AttrDict.parselist(value)
            else:
                pass

            self[key] = value

    @staticmethod
    def parselist(obj):
        l = []
        for i in obj:
            if isinstance(i, dict):
                l.append(AttrDict(i))
            elif isinstance(i, list):
                l.append(AttrDict.parselist(i))
            else:
                l.append(i)
        return l

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        elif isinstance(value, list):
            value = AttrDict.parselist(value)
        else:
            pass
        self[key] = value

    def get(self, key, default=None):
        key = key.strip().strip(".").strip()

        if key == "":
            return self
        if "." not in key:
            return getattr(self, key) if hasattr(self, key) else default
        route = key.split(".")
        obj = self
        for i, k in enumerate(route):
            if isinstance(obj, list):
                obj = obj[int(k)]
            elif isinstance(obj, AttrDict):
                obj = getattr(obj, k)
            else:
                pass
        return obj

    def set(self, key, value):
        key = key.strip().strip(".").strip()
        if key == "":
            return
        *route, key = key.split(".")
        route = ".".join(route)
        setattr(self.get(route), key, value)


def override_config(config, options=None):
    if options is None:
        return config
    for opt in options:
        assert isinstance(opt, str), f"option({opt}) should be string"
        assert ("=" in opt) and (len(opt.split("=")) == 2), f"option({opt}) should have and only have one '=' to distinguish between key and value"

        key, value = opt.split("=")
        config.set(key, eval(value))

    return config


def get_config(cfg_file, overrides=""):
    assert os.path.exists(cfg_file) and os.path.isfile(cfg_file), f"config file {cfg_file} not exists or not file!"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)
    if overrides and "=" in overrides:
        config = override_config(config, overrides.split("|"))
    return config