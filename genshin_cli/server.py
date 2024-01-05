import hydra
import uvicorn
from omegaconf import DictConfig

from genshin.api import GenshinAPI

@hydra.main(config_path="../conf", config_name="inference_config", version_base=None)
def main(cfg: DictConfig):
    app = GenshinAPI(cfg)
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=cfg.server.port if 'server' in cfg else 13245,
        root_path='/app/genshin'
    )

if __name__ == '__main__':
    main()
