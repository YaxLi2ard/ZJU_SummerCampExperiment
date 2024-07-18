import os
import random
import shutil

def delete_files_in_subfolders(root_folder, num_files_to_keep=500):
    """
    遍历根文件夹中的每个子文件夹，随机删除文件，直到每个子文件夹中只剩下指定数量的文件。

    参数:
    root_folder (str): 根文件夹的路径。
    num_files_to_keep (int): 每个子文件夹中保留的文件数量。
    """
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            num_files = len(files)  # 获取子文件夹中的文件数量
            if num_files > num_files_to_keep:
                # 随机选择需要删除的文件
                files_to_delete = random.sample(files, num_files - num_files_to_keep)
                for file in files_to_delete:
                    file_path = os.path.join(subdir_path, file)
                    try:
                        os.remove(file_path)  # 删除文件
                        print(f"删除文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件时出错: {file_path}: {e}")

def delete_subfolders(root_folder, num_folders_to_keep=100):
    """
    随机删除根文件夹中的子文件夹，直到只剩下指定数量的子文件夹。

    参数:
    root_folder (str): 根文件夹的路径。
    num_folders_to_keep (int): 根文件夹中保留的子文件夹数量。
    """
    subdirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    if len(subdirs) > num_folders_to_keep:
        # 随机选择需要删除的子文件夹
        subdirs_to_delete = random.sample(subdirs, len(subdirs) - num_folders_to_keep)
        for subdir in subdirs_to_delete:
            subdir_path = os.path.join(root_folder, subdir)
            try:
                shutil.rmtree(subdir_path)  # 删除子文件夹及其内容
                print(f"删除文件夹: {subdir_path}")
            except Exception as e:
                print(f"删除文件夹时出错: {subdir_path}: {e}")

if __name__ == "__main__":
    root_folder = './ES-imagenet-0.18/train/'
    delete_subfolders(root_folder, num_folders_to_keep=100)
    delete_files_in_subfolders(root_folder, num_files_to_keep=500)