import os
import json5
import multiprocessing as mp
from tqdm import tqdm  # 引入tqdm用于显示进度条
from ase.io import read
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile

# --- 将全局配置定义为常量 ---
ROOT_DIR = './'
ORCA_PROFILE = OrcaProfile(
    command='/home/wyk/Applications/ORCA_6.1.0/orca',
)
# --- 设置并行任务数 ---
THREADS_PER_JOB = 6
NUM_PARALLEL_JOBS = 14


# --- 将单个任务封装成一个函数 ---
def run_ocra(rid):
    """
    为单个 rid 执行完整的计算流程（优化、振动分析等）。
    """
    try:
        # 创建工作目录，如果存在，则删除，创建新的工作目录
        r_workdir = os.path.join(ROOT_DIR, 'singlepointcalc', str(rid), 'react')
        s_workdir = os.path.join(ROOT_DIR, 'singlepointcalc', str(rid), 'sp')
        p_workdir = os.path.join(ROOT_DIR, 'singlepointcalc', str(rid), 'prod')

        # 然后重新创建目录
        os.makedirs(r_workdir, exist_ok=True)
        os.makedirs(s_workdir, exist_ok=True)
        os.makedirs(p_workdir, exist_ok=True)

        # 读取原子结构
        atoms_IS = read(os.path.join(ROOT_DIR, 'dataset', f'system{rid}-react.xyz'), format='xyz')
        atoms_TS = read(os.path.join(ROOT_DIR, 'dataset', f'system{rid}-sp.xyz'), format='xyz')
        atoms_FS = read(os.path.join(ROOT_DIR, 'dataset', f'system{rid}-prod.xyz'), format='xyz')

        # 假设电荷0
        charge = 0
        num_electrons = sum(atoms_IS.get_atomic_numbers()) - charge

        # 根据电子数奇偶性设置多重度
        if num_electrons % 2 == 0:
            multiplicity = 1  # 偶数电子 -> 单重态
        else:
            multiplicity = 2  # 奇数电子 -> 双重态

        # 初始化计算器
        orcablocks = f'%pal nprocs {THREADS_PER_JOB} end'
        calc0 = ORCA(profile=ORCA_PROFILE, directory=r_workdir, charge=charge, mult=multiplicity,
                     orcasimpleinput='B3LYP def2-SVP d3 EnGrad', orcablocks=orcablocks)
        calc1 = ORCA(profile=ORCA_PROFILE, directory=s_workdir, charge=charge, mult=multiplicity,
                     orcasimpleinput='B3LYP def2-SVP d3 EnGrad', orcablocks=orcablocks)
        calc2 = ORCA(profile=ORCA_PROFILE, directory=p_workdir, charge=charge, mult=multiplicity,
                     orcasimpleinput='B3LYP def2-SVP d3 EnGrad', orcablocks=orcablocks)

        # 运行单点计算
        atoms_IS.calc = calc0
        atoms_TS.calc = calc1
        atoms_FS.calc = calc2

        for subset, image in [['react', atoms_IS], ['sp', atoms_TS], ['prod', atoms_FS]]:
            energy = image.get_potential_energy()
            forces = image.get_forces(apply_constraint=False)

            _res = {
                "name": f'system{rid}',
                "subset": subset,
                "atomic_number": image.get_atomic_numbers().tolist(),
                "charge": [0] * len(image),
                "positition": image.get_positions().tolist(),
                "energies": float(energy),
                "forces": forces.tolist(),
            }

            # 写入结果文件
            with open(os.path.join(ROOT_DIR, 'result', f'{rid}_{subset}.json'), 'w') as f:
                f.write(json5.dumps(_res, quote_keys=True, trailing_commas=False, indent=2))

        return rid, "Success"

    except Exception as e:
        # 捕获任何在计算中发生的错误
        error_message = f"!!! Job for rid={rid} FAILED with error: {e}"
        # print(error_message)
        # 可以选择将错误信息写入一个日志文件
        with open(os.path.join(ROOT_DIR, 'error_log.txt'), 'a') as f:
            f.write(error_message + '\n')
        return rid, "Failed"


# ---  主程序入口 ---
if __name__ == '__main__':
    # 定义要处理的任务列表
    rids_to_process = range(3, 122)

    # 打印执行信息
    print("--- Starting parallel ORCA calculation ---")
    print(f" - Starting parallel execution for {len(rids_to_process)} jobs.")
    print(f" - Using {NUM_PARALLEL_JOBS} parallel processes.")
    print(f" - Each ORCA job will use {THREADS_PER_JOB} cores.")
    print(f" - Total cores: {THREADS_PER_JOB * min(NUM_PARALLEL_JOBS, len(rids_to_process))}\n")

    # 创建并运行进程池
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=min(NUM_PARALLEL_JOBS, len(rids_to_process))) as pool:
        # 使用 pool.imap_unordered 和 tqdm 来获取实时进度
        # imap_unordered 会在任务完成时立即返回结果，而不是等待所有任务完成
        results = list(tqdm(
            pool.imap_unordered(run_ocra, rids_to_process),
            total=len(rids_to_process),
            desc="Running···",
            colour="green"
        ))

    print("\n--- All tasks completed. ---")

    # 打印执行结果总结
    success_count = sum(1 for r, status in results if status == "Success")
    failed_count = len(results) - success_count
    print(f"Summary: {success_count} jobs succeeded, {failed_count} jobs failed.")
    if failed_count > 0:
        print("Check 'error_log.txt' for details on failed jobs.")
