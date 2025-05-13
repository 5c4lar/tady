import hydra
import shutil
import pathlib
import subprocess

def configure(cfg):
    work_dir = pathlib.Path(cfg.build_dir)
    configure = cfg.project.configure32 if cfg.compiler.mode == 32 else cfg.project.configure64
    env = cfg.compiler.envs
    work_dir.mkdir(parents=True, exist_ok=True)
    install_dir = pathlib.Path(cfg.install_dir)
    if install_dir.exists():
        return False
    configure = f"{configure} --prefix={install_dir}"
    env["CFLAGS"] += " " + " ".join(cfg.opt.options)
    env["CXXFLAGS"] += " " + " ".join(cfg.opt.options)
    print(env)
    res = subprocess.run(configure, cwd=work_dir, env=env, shell=True)
    return res.returncode == 0
    
def install(cfg):
    work_dir = pathlib.Path(cfg.build_dir)
    subprocess.run(f"make install -j{cfg.jobs}", cwd=work_dir, shell=True)
    shutil.rmtree(work_dir)
    
@hydra.main(version_base=None, config_path="conf", config_name="compile.yaml")
def main(cfg):
    ret = configure(cfg)
    if ret:
        install(cfg)

if __name__ == "__main__":
    main()
