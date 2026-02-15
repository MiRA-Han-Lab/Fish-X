import os
import numpy as np
from pathlib import Path
import glob

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_swc_file(filepath):
    """加载SWC文件"""
    return np.loadtxt(filepath)

def save_swc_file(filepath, data, format_type='float'):
    """保存SWC文件
    
    Args:
        filepath: 文件路径
        data: 数据数组
        format_type: 'float' 为普通数值格式，'swc' 为SWC格式
    """
    if format_type == 'swc':
        # SWC格式：前两列为整数，中间三列为浮点数，最后两列为整数
        with open(filepath, 'w') as f:
            for row in data:
                f.write(f"{int(row[0])} {int(row[1])} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f} {int(row[5])} {int(row[6])}\n")
    else:
        # 普通浮点数格式
        np.savetxt(filepath, data)

def process_downsampling(source_path，dst_path):
    """第一部分：下采样处理"""
    print("开始下采样处理...")    
    ensure_dir(dst_path)
    
    # 获取所有SWC文件
    swc_files = glob.glob(os.path.join(source_path, '*.swc'))
    
    for file_path in swc_files:
        filename = os.path.basename(file_path)
        dst_file = os.path.join(dst_path, filename)
 
        
        # 加载数据并提取坐标列（第3-5列，索引2-4）
        swc_data = load_swc_file(file_path)
        swc_coords = swc_data[:, 2:5]  # 提取x,y,z坐标
        
        # 进行坐标变换
        swc_x_y = swc_coords[:, :2] / 248  # x,y坐标除以248
        swc_z = (swc_coords[:, 2] - 2953) / 30  # z坐标变换
        
        # 组合变换后的坐标
        transformed_coords = np.column_stack([swc_x_y, swc_z])
        
        # 保存结果
        save_swc_file(dst_file, transformed_coords)
        print(f"处理完成: {filename}")

def process_matrix_transformation(source_path,dst_path,M):
    """第二部分：矩阵变换处理"""
    
    ensure_dir(dst_path)
    
    # 获取所有SWC文件
    swc_files = glob.glob(os.path.join(source_path, '*.swc'))
    
    for file_path in swc_files:
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        dst_file = os.path.join(dst_path, f"{name_without_ext}_r.swc")
        
        # 加载数据
        swc_data = load_swc_file(file_path)
        
        # 添加齐次坐标列（全为1）
        ones_column = np.ones((swc_data.shape[0], 1))
        swc_homogeneous = np.column_stack([swc_data, ones_column])
        
        # 矩阵变换
        transformed = (M @ swc_homogeneous.T).T
        
        # z坐标加135
        transformed[:, 2] += 135
        
        # 保存前三列（x,y,z坐标）
        save_swc_file(dst_file, transformed[:, :3])
        print(f"矩阵变换完成: {filename}")

def process_coordinate_transformation(source_path,dst_path):
    """第三部分：坐标变换处理"""
    ensure_dir(dst_path)
    
    # 获取所有SWC文件
    swc_files = glob.glob(os.path.join(source_path, '*.swc'))
    
    for file_path in swc_files:
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        result_file = os.path.join(result_path, f"{name_without_ext}_r.swc")
        dst_file = os.path.join(dst_path, f"{name_without_ext}_r_r.swc")
        
        # 加载原始数据和变换后的坐标数据
        original_data = load_swc_file(file_path)
        transformed_coords = load_swc_file(result_file)
        
        # 坐标变换
        tmp = 1035 - transformed_coords[:, 2]
        transformed_coords[:, 2] = 382 - transformed_coords[:, 1]
        transformed_coords[:, 1] = tmp
        
        # 替换原始数据的坐标列（第3-5列）
        original_data[:, 2:5] = transformed_coords
        
        # 保存为SWC格式
        save_swc_file(dst_file, original_data, format_type='swc')
        print(f"坐标变换完成: {filename}")

def process_final_adjustment(source_path,dst_path):

    ensure_dir(dst_path)
    
    # 获取所有SWC文件
    swc_files = glob.glob(os.path.join(source_path, '*.swc'))
    
    for file_path in swc_files:
        filename = os.path.basename(file_path)
        dst_file = os.path.join(dst_path, filename)
        
        # 加载数据
        swc_data = load_swc_file(file_path)
        
        # 设置第6列为1（索引5）
        swc_data[:, 5] = 1
        
        # 保存为SWC格式
        save_swc_file(dst_file, swc_data, format_type='swc')
        print(f"最终调整完成: {filename}")

def main():
    """主函数：执行所有处理步骤"""
    print("开始SWC文件处理流程...")
    
    try:
        # 执行所有处理步骤
        process_downsampling(source_path,dst_path)
        # process_matrix_transformation(source_path，dst_path,M)
        # process_coordinate_transformation(source_path，dst_path)
        # process_final_adjustment(source_path，dst_path)
        
        print("\n所有处理步骤完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()