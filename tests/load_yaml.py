from utils.helpers import load_yaml_config

def test_load_tiny_yaml():
    config = load_yaml_config('config/tiny.yaml')
    assert isinstance(config, dict), "Config should be a dictionary"
    assert 'host_configurations' in config, "'host_configurations' key missing in config"

if __name__ == "__main__":
    test_load_tiny_yaml()
    print("Config loading test passed.")