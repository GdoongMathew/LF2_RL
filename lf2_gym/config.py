import configparser
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Lf2GymConfig:
    exe: Path
    mode: str
    fighter: str
    fighter_id: int
    team: str
    background: str
    difficulty: str
    com_player_num: int = 4

    @property
    def control_txt(self) -> Path:
        parent = self.exe.parent
        control_txt = parent / "data" / "control.txt"
        assert control_txt.exists()
        return control_txt


def _read_config() -> Lf2GymConfig:
    config = configparser.ConfigParser()
    init_path = Path(__file__).parent / "lf2_envs" / "config" / "config.ini"
    config.read(init_path)
    lf2_config = dict(**config["lf2_config"])
    lf2_config["exe"] = Path(lf2_config["exe"])
    lf2_config["fighter_id"] = int(lf2_config["fighter_id"])
    lf2_config["com_player_num"] = int(lf2_config["com_player_num"])
    return Lf2GymConfig(**lf2_config)


LF2_GYM_CONFIG = _read_config()