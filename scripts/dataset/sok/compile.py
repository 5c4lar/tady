import hydra
import shutil
import pathlib
import subprocess
import os

def configure(cfg):
    work_dir = pathlib.Path(cfg.build_dir)
    configure = cfg.project.configure32 if cfg.compiler.mode == 32 else cfg.project.configure64
    env = os.environ.copy()
    env.update(cfg.compiler.envs)
    work_dir.mkdir(parents=True, exist_ok=True)
    install_dir = pathlib.Path(cfg.install_dir)
    if install_dir.exists():
        return False
    if cfg.project.type == "autotools":
        configure = f"{configure} --prefix={install_dir}"
    elif cfg.project.type == "cmake":
        configure = f"cmake {configure} -DCMAKE_INSTALL_PREFIX={install_dir}"
    elif cfg.project.type == "makefile":
        configure = f"{configure} {work_dir}"
    env["CFLAGS"] += " " + " ".join(cfg.opt.options)
    env["CXXFLAGS"] += " " + " ".join(cfg.opt.options)
    res = subprocess.run(configure, cwd=work_dir, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0
    
def install(cfg):
    work_dir = pathlib.Path(cfg.build_dir)
    env = os.environ.copy()
    env.update(cfg.compiler.envs)
    env["CFLAGS"] += " " + " ".join(cfg.opt.options)
    env["CXXFLAGS"] += " " + " ".join(cfg.opt.options)
    if cfg.project.type == "makefile":
        install_dir = pathlib.Path(cfg.install_dir)
        install_dir.mkdir(parents=True, exist_ok=True)
        ret = subprocess.run(f"make -j{cfg.jobs} {cfg.project.target}", cwd=work_dir, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            print(ret.stderr)
            return False
        subprocess.run(f"cp -r {work_dir}/* {cfg.install_dir}", cwd=work_dir, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        ret = subprocess.run(f"make -j{cfg.jobs}", cwd=work_dir, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode != 0:
            print(ret.stderr)
            return False
        subprocess.run(f"make install -j{cfg.jobs}", cwd=work_dir, env=env, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    shutil.rmtree(work_dir)
    
@hydra.main(version_base=None, config_path="conf", config_name="compile.yaml")
def main(cfg):
    ret = configure(cfg)
    if ret:
        install(cfg)

if __name__ == "__main__":
    main()
